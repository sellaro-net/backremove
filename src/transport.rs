use axum::serve::Listener;
use std::{
    future::Future,
    io::{self, IoSlice},
    mem::MaybeUninit,
    net::SocketAddr,
    pin::Pin,
    task::{Context, Poll, ready},
    time::Duration,
};
use tokio::{
    io::{AsyncRead, AsyncWrite, ReadBuf},
    net::{TcpListener, TcpStream},
    time::Sleep,
};

const CLOSE_GRACE: Duration = Duration::from_secs(1);
const DRAIN_BUFFER_BYTES: usize = 8 * 1024;
const DRAIN_READS_PER_POLL: usize = 16;

/// Keeps a rejected upload's TCP read half alive briefly after sending the response.
///
/// Only transport shutdown drains input; HTTP request bodies remain untouched.
/// After shutting down writes, each connection drains at most `max_drain_bytes`
/// for at most one second, ending sooner on peer EOF or a read error. Reaching
/// either bound permits closing even if the peer still has unread input.
#[derive(Debug)]
pub struct GracefulTcpListener {
    listener: TcpListener,
    max_drain_bytes: usize,
}

impl GracefulTcpListener {
    pub fn new(listener: TcpListener, max_drain_bytes: usize) -> Self {
        Self {
            listener,
            max_drain_bytes,
        }
    }
}

impl Listener for GracefulTcpListener {
    type Io = GracefulTcpStream;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        // Preserve Axum's TCP accept-error logging, retry and backoff policy.
        let (stream, remote_addr) = Listener::accept(&mut self.listener).await;
        (
            GracefulTcpStream {
                stream,
                max_drain_bytes: self.max_drain_bytes,
                shutdown: Shutdown::Open,
            },
            remote_addr,
        )
    }

    fn local_addr(&self) -> io::Result<Self::Addr> {
        self.listener.local_addr()
    }
}

#[derive(Debug)]
pub struct GracefulTcpStream {
    stream: TcpStream,
    max_drain_bytes: usize,
    shutdown: Shutdown,
}

#[derive(Debug)]
enum Shutdown {
    Open,
    Draining {
        remaining: usize,
        deadline: Pin<Box<Sleep>>,
    },
    Closed,
}

impl AsyncRead for GracefulTcpStream {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().stream).poll_read(cx, buf)
    }
}

impl AsyncWrite for GracefulTcpStream {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.get_mut().stream).poll_write(cx, buf)
    }

    fn poll_write_vectored(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.get_mut().stream).poll_write_vectored(cx, bufs)
    }

    fn is_write_vectored(&self) -> bool {
        self.stream.is_write_vectored()
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().stream).poll_flush(cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        if matches!(this.shutdown, Shutdown::Open) {
            // Hyper has flushed the response. Send FIN before waiting for input,
            // so the client can observe its rejection without finishing upload.
            if let Err(error) = ready!(Pin::new(&mut this.stream).poll_shutdown(cx)) {
                this.shutdown = Shutdown::Closed;
                return Poll::Ready(Err(error));
            }
            if this.max_drain_bytes == 0 {
                this.shutdown = Shutdown::Closed;
            } else {
                this.shutdown = Shutdown::Draining {
                    remaining: this.max_drain_bytes,
                    deadline: Box::pin(tokio::time::sleep(CLOSE_GRACE)),
                };
            }
        }

        let Shutdown::Draining {
            remaining,
            deadline,
        } = &mut this.shutdown
        else {
            return Poll::Ready(Ok(()));
        };
        let mut scratch = [MaybeUninit::uninit(); DRAIN_BUFFER_BYTES];
        for _ in 0..DRAIN_READS_PER_POLL {
            // Register the timer even when the peer is silent, and check it
            // between reads so a continuously writable peer cannot extend it.
            if *remaining == 0 || deadline.as_mut().poll(cx).is_ready() {
                this.shutdown = Shutdown::Closed;
                return Poll::Ready(Ok(()));
            }
            let capacity = (*remaining).min(scratch.len());
            let mut buf = ReadBuf::uninit(&mut scratch[..capacity]);
            match Pin::new(&mut this.stream).poll_read(cx, &mut buf) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Ok(())) if !buf.filled().is_empty() => {
                    *remaining -= buf.filled().len();
                }
                // The response was already sent. EOF and reset both finish
                // this best-effort raw drain, not an HTTP body read.
                Poll::Ready(_) => {
                    this.shutdown = Shutdown::Closed;
                    return Poll::Ready(Ok(()));
                }
            }
        }

        // Limit one poll to 128 KiB even if TCP is always readable. No spawned
        // drain task survives the connection or the server's shutdown future.
        cx.waker().wake_by_ref();
        Poll::Pending
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Router,
        extract::{ConnectInfo, Request},
        http::StatusCode,
        routing::post,
        serve::IncomingStream,
    };
    use std::future::IntoFuture;
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        sync::{mpsc, oneshot},
        task::JoinHandle,
        time::{sleep, timeout},
    };
    use tower::Service;

    struct RejectingServer {
        address: SocketAddr,
        rejected: mpsc::Receiver<SocketAddr>,
        shutdown: oneshot::Sender<()>,
        task: JoinHandle<io::Result<()>>,
    }

    async fn rejecting_server(max_drain_bytes: usize) -> RejectingServer {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (rejected_tx, rejected) = mpsc::channel(1);
        let router = Router::new().route(
            "/",
            post(
                move |ConnectInfo(peer): ConnectInfo<SocketAddr>, _request: Request| {
                    let rejected_tx = rejected_tx.clone();
                    async move {
                        // Deliberately never poll the request body, just as an
                        // admission rejection returns before reading multipart.
                        rejected_tx.send(peer).await.unwrap();
                        (StatusCode::SERVICE_UNAVAILABLE, "busy")
                    }
                },
            ),
        );
        let mut make_service = router.into_make_service_with_connect_info::<SocketAddr>();
        let make_service =
            tower::service_fn(move |incoming: IncomingStream<'_, GracefulTcpListener>| {
                make_service.call(*incoming.remote_addr())
            });
        let (shutdown, stopped) = oneshot::channel();
        let task = tokio::spawn(
            axum::serve(
                GracefulTcpListener::new(listener, max_drain_bytes),
                make_service,
            )
            .with_graceful_shutdown(async {
                let _ = stopped.await;
            })
            .into_future(),
        );
        RejectingServer {
            address,
            rejected,
            shutdown,
            task,
        }
    }

    async fn send_headers(client: &mut TcpStream, content_length: usize) {
        client
            .write_all(
                format!(
                    "POST / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Length: {content_length}\r\n\r\n"
                )
                .as_bytes(),
            )
            .await
            .unwrap();
    }

    async fn read_rejection(client: &mut TcpStream) {
        let mut response = Vec::new();
        timeout(Duration::from_secs(2), client.read_to_end(&mut response))
            .await
            .expect("response and server FIN must precede upload completion")
            .expect("rejection must not be replaced by a TCP reset");
        let response = String::from_utf8(response).unwrap();
        assert!(response.starts_with("HTTP/1.1 503 "), "{response}");
        assert!(response.ends_with("\r\n\r\nbusy"), "{response}");
    }

    #[tokio::test]
    async fn delayed_rejected_upload_receives_503_without_reset() {
        let mut server = rejecting_server(128 * 1024).await;
        let mut client = TcpStream::connect(server.address).await.unwrap();
        client.set_nodelay(true).unwrap();
        send_headers(&mut client, 64 * 1024).await;
        assert_eq!(
            timeout(Duration::from_secs(2), server.rejected.recv())
                .await
                .unwrap()
                .unwrap(),
            client.local_addr().unwrap()
        );
        server.shutdown.send(()).unwrap();
        sleep(Duration::from_millis(3)).await;
        let closed_before_upload = server.task.is_finished();
        timeout(Duration::from_secs(2), client.write_all(&[b'x'; 64 * 1024]))
            .await
            .unwrap()
            .expect("a late upload must not race an immediate socket close");
        client.shutdown().await.unwrap();
        read_rejection(&mut client).await;
        timeout(Duration::from_millis(500), server.task)
            .await
            .expect("peer EOF must finish the drain before its deadline")
            .unwrap()
            .unwrap();
        // This also detects the original immediate close on platforms where a
        // late write happens to succeed before the peer's reset is reported.
        assert!(
            !closed_before_upload,
            "connection closed before the late body"
        );
    }

    #[tokio::test]
    async fn silent_upload_peer_keeps_read_half_until_finite_close_deadline() {
        let mut server = rejecting_server(128 * 1024).await;
        let mut client = TcpStream::connect(server.address).await.unwrap();
        send_headers(&mut client, 64 * 1024).await;
        timeout(Duration::from_secs(2), server.rejected.recv())
            .await
            .unwrap()
            .unwrap();
        read_rejection(&mut client).await;
        server.shutdown.send(()).unwrap();

        // Receiving the response and FIN must not tear down the read half. The
        // unwrapped TcpListener completes server shutdown immediately here.
        assert!(
            timeout(Duration::from_millis(100), &mut server.task)
                .await
                .is_err(),
            "the still-open upload peer must get its drain grace"
        );
        timeout(CLOSE_GRACE * 2, server.task)
            .await
            .expect("a peer that never writes or closes must not hold shutdown")
            .unwrap()
            .unwrap();
        // Keep the client's write half open throughout both deadline checks.
        drop(client);
    }

    #[tokio::test]
    async fn drain_byte_limit_releases_connection_without_peer_eof() {
        const LIMIT: usize = 16 * 1024;
        let mut server = rejecting_server(LIMIT).await;
        let mut client = TcpStream::connect(server.address).await.unwrap();
        send_headers(&mut client, LIMIT * 2).await;
        timeout(Duration::from_secs(2), server.rejected.recv())
            .await
            .unwrap()
            .unwrap();
        read_rejection(&mut client).await;
        server.shutdown.send(()).unwrap();

        // FIN proves shutdown has begun, so these bytes bypass HTTP entirely.
        client.write_all(&[b'x'; LIMIT - 1]).await.unwrap();
        assert!(
            timeout(Duration::from_millis(100), &mut server.task)
                .await
                .is_err(),
            "drain must not close before EOF, its deadline or its byte limit"
        );
        client.write_all(b"x").await.unwrap();
        timeout(Duration::from_millis(500), server.task)
            .await
            .expect("the exact byte limit must release a still-open peer")
            .unwrap()
            .unwrap();
        drop(client);
    }
}
