use crate::{
    config::Config,
    error::{AppError, Result},
    image_pipeline::ImagePipeline,
    inference::InferenceEngine,
    types::{
        DecodedImage, Mask, MaskGeometry, Model, ModelSpec, PreparedImage, Timings, WorkClass,
    },
};
use bytes::Bytes;
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Condvar, Mutex, MutexGuard, Weak,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore, oneshot};

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

/// A byte reservation is released by its actual last owner, never by a timeout waiter.
struct Budget {
    limit: usize,
    used: Mutex<usize>,
    changed: Condvar,
}
impl Budget {
    fn new(limit: usize) -> Arc<Self> {
        Arc::new(Self {
            limit,
            used: Mutex::new(0),
            changed: Condvar::new(),
        })
    }
    fn try_acquire(self: &Arc<Self>, bytes: usize) -> Option<MemoryLease> {
        let mut used = lock(&self.used);
        let next = used.checked_add(bytes)?;
        if next > self.limit {
            return None;
        }
        *used = next;
        Some(MemoryLease {
            budget: self.clone(),
            bytes,
        })
    }
    fn acquire(self: &Arc<Self>, bytes: usize, job: &Job, shared: &Shared) -> Result<MemoryLease> {
        if bytes > self.limit {
            return Err(AppError::TooLarge);
        }
        let mut used = lock(&self.used);
        loop {
            if shared.failed.load(Ordering::Acquire) {
                return Err(AppError::Unavailable);
            }
            job.check()?;
            if let Some(next) = used.checked_add(bytes).filter(|next| *next <= self.limit) {
                *used = next;
                return Ok(MemoryLease {
                    budget: self.clone(),
                    bytes,
                });
            }
            let remaining = job.deadline.saturating_duration_since(Instant::now());
            let (guard, _) = self
                .changed
                .wait_timeout(used, remaining)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            used = guard;
        }
    }
    fn used(&self) -> usize {
        *lock(&self.used)
    }
}
struct MemoryLease {
    budget: Arc<Budget>,
    bytes: usize,
}
impl MemoryLease {
    fn shrink(&mut self, bytes: usize) -> Result<()> {
        if bytes > self.bytes {
            return Err(AppError::Internal);
        }
        let released = self.bytes - bytes;
        *lock(&self.budget.used) -= released;
        self.bytes = bytes;
        self.budget.changed.notify_all();
        Ok(())
    }
}
impl Drop for MemoryLease {
    fn drop(&mut self) {
        *lock(&self.budget.used) -= self.bytes;
        self.budget.changed.notify_all();
    }
}

pub(crate) struct Admission {
    owner: Weak<Shared>,
    _slot: OwnedSemaphorePermit,
    input: Option<MemoryLease>,
}

pub(crate) struct Submission {
    pub input: Vec<u8>,
    pub content_type: String,
    pub model: Model,
    pub class: WorkClass,
    pub started: Instant,
    pub deadline: Instant,
}

struct PriorityQueue<T> {
    foreground: VecDeque<T>,
    background: VecDeque<T>,
    foreground_burst: u8,
}
impl<T> Default for PriorityQueue<T> {
    fn default() -> Self {
        Self {
            foreground: VecDeque::new(),
            background: VecDeque::new(),
            foreground_burst: 0,
        }
    }
}
impl<T> PriorityQueue<T> {
    fn len(&self) -> usize {
        self.foreground.len() + self.background.len()
    }
    fn push(&mut self, class: WorkClass, job: T) {
        match class {
            WorkClass::Foreground => self.foreground.push_back(job),
            WorkClass::Background => self.background.push_back(job),
        }
    }
    fn pop(&mut self) -> Option<T> {
        if !self.background.is_empty() && (self.foreground.is_empty() || self.foreground_burst >= 3)
        {
            self.foreground_burst = 0;
            self.background.pop_front()
        } else if let Some(job) = self.foreground.pop_front() {
            self.foreground_burst = if self.background.is_empty() {
                0
            } else {
                self.foreground_burst + 1
            };
            Some(job)
        } else {
            None
        }
    }
}

struct Control {
    cancelled: AtomicBool,
    started: AtomicBool,
}
struct Job {
    control: Arc<Control>,
    reply: oneshot::Sender<Result<CompletedJob>>,
    admission: Admission,
    input: Vec<u8>,
    content_type: String,
    spec: ModelSpec,
    class: WorkClass,
    deadline: Instant,
    enqueued: Instant,
    timings: Timings,
}
impl Job {
    fn deadline_error(&self) -> AppError {
        if self.control.started.load(Ordering::Acquire) {
            AppError::Deadline
        } else {
            AppError::Busy
        }
    }
    fn check(&self) -> Result<()> {
        if self.control.cancelled.load(Ordering::Acquire)
            || self.reply.is_closed()
            || Instant::now() >= self.deadline
        {
            Err(self.deadline_error())
        } else {
            Ok(())
        }
    }
    fn fail(self, error: AppError) {
        let _ = self.reply.send(Err(error));
    }
}

pub(crate) struct CompletedJob {
    pub body: Bytes,
    pub model: Model,
    pub timings: Timings,
}
struct ResponseOwner {
    bytes: Vec<u8>,
    _memory: MemoryLease,
    _admission: Admission,
}
impl AsRef<[u8]> for ResponseOwner {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}

pub(crate) struct JobTicket {
    control: Arc<Control>,
    receiver: oneshot::Receiver<Result<CompletedJob>>,
    shared: Arc<Shared>,
    deadline: Instant,
}
impl JobTicket {
    pub async fn wait(mut self) -> Result<CompletedJob> {
        let deadline = tokio::time::Instant::from_std(self.deadline);
        let result = tokio::select! {
            biased;
            result = &mut self.receiver => result.unwrap_or(Err(AppError::Internal)),
            _ = tokio::time::sleep_until(deadline) => Err(self.deadline_error()),
        };
        if result.is_ok() && Instant::now() >= self.deadline {
            return Err(self.deadline_error());
        }
        result
    }
    fn deadline_error(&self) -> AppError {
        if self.control.started.load(Ordering::Acquire) {
            AppError::Deadline
        } else {
            AppError::Busy
        }
    }
}
impl Drop for JobTicket {
    fn drop(&mut self) {
        self.control.cancelled.store(true, Ordering::Release);
        // Removing queued work here releases upload/admission reservations promptly,
        // even while the dedicated model thread is still occupied by another request.
        let mut queue = lock(&self.shared.queue);
        queue
            .foreground
            .retain(|job| !Arc::ptr_eq(&job.control, &self.control));
        queue
            .background
            .retain(|job| !Arc::ptr_eq(&job.control, &self.control));
        drop(queue);
        self.shared.changed.notify_all();
        self.shared.memory.changed.notify_all();
    }
}

struct Shared {
    config: Arc<Config>,
    queue: Mutex<PriorityQueue<Job>>,
    changed: Condvar,
    admission: Arc<Semaphore>,
    input: Arc<Budget>,
    memory: Arc<Budget>,
    stopping: AtomicBool,
    failed: AtomicBool,
    decoding: AtomicUsize,
    prepared: AtomicUsize,
    inferring: AtomicUsize,
    encoding: AtomicUsize,
    completed: AtomicU64,
    live_workers: AtomicUsize,
    workers_done: Notify,
    handles: Mutex<Vec<JoinHandle<()>>>,
    born: Instant,
    last_progress: AtomicU64,
    inference_started: AtomicU64,
    inference_deadline: AtomicU64,
    engine_alive: AtomicBool,
    fast: ModelSpec,
    quality: Option<ModelSpec>,
    model_status: Value,
    provider: String,
}
impl Shared {
    fn progress(&self) {
        self.last_progress
            .store(self.born.elapsed().as_millis() as u64, Ordering::Release);
    }
    fn fail(&self, worker: &'static str) {
        tracing::error!(
            worker,
            "Ein Verarbeitungsworker ist ausgefallen; keine weitere Annahme."
        );
        self.failed.store(true, Ordering::Release);
        self.stopping.store(true, Ordering::Release);
        self.admission.close();
        let mut queue = lock(&self.queue);
        while let Some(job) = queue.pop() {
            job.fail(AppError::Unavailable);
        }
        drop(queue);
        self.changed.notify_all();
        self.memory.changed.notify_all();
    }
    fn next_job(&self) -> Option<Job> {
        let mut queue = lock(&self.queue);
        loop {
            if self.failed.load(Ordering::Acquire) {
                return None;
            }
            if let Some(job) = queue.pop() {
                if let Err(error) = job.check() {
                    job.fail(error);
                    continue;
                }
                return Some(job);
            }
            if self.stopping.load(Ordering::Acquire) {
                return None;
            }
            queue = self
                .changed
                .wait(queue)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }
}

#[derive(Clone)]
pub struct Scheduler {
    shared: Arc<Shared>,
}
impl Scheduler {
    pub fn start(
        config: Arc<Config>,
        pipeline: Arc<ImagePipeline>,
        engine: InferenceEngine,
    ) -> Result<Self> {
        let fast = engine.spec(Model::Fast).ok_or(AppError::Unavailable)?;
        let shared = Arc::new(Shared {
            queue: Mutex::new(PriorityQueue::default()),
            changed: Condvar::new(),
            admission: Arc::new(Semaphore::new(config.max_jobs)),
            input: Budget::new(config.input_budget_bytes),
            memory: Budget::new(config.memory_budget_bytes),
            stopping: AtomicBool::new(false),
            failed: AtomicBool::new(false),
            decoding: AtomicUsize::new(0),
            prepared: AtomicUsize::new(0),
            inferring: AtomicUsize::new(0),
            encoding: AtomicUsize::new(0),
            completed: AtomicU64::new(0),
            live_workers: AtomicUsize::new(0),
            workers_done: Notify::new(),
            handles: Mutex::new(Vec::new()),
            born: Instant::now(),
            last_progress: AtomicU64::new(0),
            inference_started: AtomicU64::new(0),
            inference_deadline: AtomicU64::new(0),
            engine_alive: AtomicBool::new(true),
            fast,
            quality: engine.spec(Model::Quality),
            model_status: engine.status(),
            provider: engine.provider().to_owned(),
            config,
        });
        let scheduler = Self {
            shared: shared.clone(),
        };
        let (prepared_tx, prepared_rx) = flume::bounded(1);
        let (prepare_slot_tx, prepare_slot_rx) = flume::bounded(1);
        prepare_slot_tx.send(()).map_err(|_| AppError::Internal)?;
        let (encode_tx, encode_rx) = flume::bounded(2);
        let (encode_slot_tx, encode_slot_rx) = flume::bounded(2);
        for _ in 0..2 {
            encode_slot_tx.send(()).map_err(|_| AppError::Internal)?;
        }
        let result = (|| {
            for _ in 0..2 {
                let worker_shared = shared.clone();
                let worker_pipeline = pipeline.clone();
                let receiver = encode_rx.clone();
                spawn_worker(&shared, "backremove-encode", move || {
                    encode_loop(worker_shared, worker_pipeline, receiver)
                })?;
            }
            let worker_shared = shared.clone();
            spawn_worker(&shared, "backremove-inference", move || {
                inference_loop(
                    worker_shared,
                    engine,
                    prepared_rx,
                    encode_tx,
                    encode_slot_rx,
                    encode_slot_tx,
                )
            })?;
            let worker_shared = shared.clone();
            spawn_worker(&shared, "backremove-prepare", move || {
                prepare_loop(
                    worker_shared,
                    pipeline,
                    prepared_tx,
                    prepare_slot_rx,
                    prepare_slot_tx,
                )
            })?;
            Ok(())
        })();
        if let Err(error) = result {
            shared.fail("startup");
            return Err(error);
        }
        Ok(scheduler)
    }
    pub(crate) fn try_admit(&self, model: Model) -> Result<Admission> {
        if !self.ready() {
            return Err(AppError::Unavailable);
        }
        if model == Model::Quality && self.shared.quality.is_none() {
            return Err(AppError::Unavailable);
        }
        let slot = self
            .shared
            .admission
            .clone()
            .try_acquire_owned()
            .map_err(|_| AppError::Busy)?;
        // Multipart buffering may coexist with the contiguous file buffer.
        let bytes = self
            .shared
            .config
            .image
            .max_multipart_bytes
            .checked_mul(2)
            .ok_or(AppError::Internal)?;
        let input = self.shared.input.try_acquire(bytes).ok_or(AppError::Busy)?;
        Ok(Admission {
            owner: Arc::downgrade(&self.shared),
            _slot: slot,
            input: Some(input),
        })
    }
    pub(crate) fn submit(&self, mut admission: Admission, submission: Submission) -> JobTicket {
        let Submission {
            input,
            content_type,
            model,
            class,
            started,
            deadline,
        } = submission;
        let (reply, receiver) = oneshot::channel();
        let control = Arc::new(Control {
            cancelled: AtomicBool::new(false),
            started: AtomicBool::new(false),
        });
        let ticket = JobTicket {
            control: control.clone(),
            receiver,
            shared: self.shared.clone(),
            deadline,
        };
        let spec = match model {
            Model::Fast => Some(self.shared.fast.clone()),
            Model::Quality => self.shared.quality.clone(),
        };
        let Some(spec) = spec else {
            let _ = reply.send(Err(AppError::Unavailable));
            return ticket;
        };
        if !Weak::ptr_eq(&admission.owner, &Arc::downgrade(&self.shared))
            || self.shared.stopping.load(Ordering::Acquire)
        {
            let _ = reply.send(Err(AppError::Unavailable));
            return ticket;
        }
        if input.is_empty() || input.len() > self.shared.config.image.max_file_bytes {
            let _ = reply.send(Err(if input.is_empty() {
                AppError::InvalidImage
            } else {
                AppError::TooLarge
            }));
            return ticket;
        }
        // Multipart and transport buffers are gone. Queued work retains only
        // its actual contiguous input capacity, not the peak upload allowance.
        let reservation = admission
            .input
            .as_mut()
            .ok_or(AppError::Internal)
            .and_then(|lease| lease.shrink(input.capacity()));
        if let Err(error) = reservation {
            let _ = reply.send(Err(error));
            return ticket;
        }
        let enqueued = Instant::now();
        if enqueued >= deadline {
            let _ = reply.send(Err(AppError::Busy));
            return ticket;
        }
        let job = Job {
            control,
            reply,
            admission,
            input,
            content_type,
            spec,
            class,
            deadline,
            enqueued,
            timings: Timings {
                admission_ms: millis(enqueued.saturating_duration_since(started)),
                ..Timings::default()
            },
        };
        let mut queue = lock(&self.shared.queue);
        if self.shared.stopping.load(Ordering::Acquire) {
            job.fail(AppError::Unavailable);
        } else if queue.len() >= self.shared.config.queue_capacity {
            job.fail(AppError::Busy);
        } else {
            queue.push(class, job);
        }
        drop(queue);
        self.shared.changed.notify_one();
        ticket
    }
    pub fn ready(&self) -> bool {
        if self.shared.stopping.load(Ordering::Acquire)
            || self.shared.failed.load(Ordering::Acquire)
        {
            return false;
        }
        let active_deadline = self.shared.inference_deadline.load(Ordering::Acquire);
        active_deadline == 0 || self.shared.born.elapsed().as_millis() < u128::from(active_deadline)
    }
    pub fn provider(&self) -> &str {
        &self.shared.provider
    }
    pub fn model_status(&self) -> Value {
        let mut models = self.shared.model_status.clone();
        if !self.ready() {
            let alive = self.shared.engine_alive.load(Ordering::Acquire);
            let status = if self.shared.failed.load(Ordering::Acquire) {
                "failed"
            } else if self.shared.stopping.load(Ordering::Acquire) {
                "draining"
            } else {
                "deadline_exceeded"
            };
            if let Some(entries) = models.as_object_mut() {
                for entry in entries.values_mut().filter(|entry| entry["loaded"] == true) {
                    entry["ready"] = json!(false);
                    entry["loaded"] = json!(alive);
                    entry["status"] = json!(status);
                }
            }
        }
        models
    }
    pub fn status(&self) -> Value {
        let queue = lock(&self.shared.queue);
        let tick = self.shared.born.elapsed().as_millis() as u64;
        let inference_started = self.shared.inference_started.load(Ordering::Acquire);
        json!({
            "ready": self.ready(), "stopping": self.shared.stopping.load(Ordering::Acquire), "failed": self.shared.failed.load(Ordering::Acquire),
            "capacity": self.shared.config.queue_capacity, "max_jobs": self.shared.config.max_jobs,
            "accepted": self.shared.config.max_jobs - self.shared.admission.available_permits(),
            "waiting": queue.len(), "foreground": queue.foreground.len(), "background": queue.background.len(),
            "decoding": self.shared.decoding.load(Ordering::Acquire), "prepared": self.shared.prepared.load(Ordering::Acquire),
            "inferring": self.shared.inferring.load(Ordering::Acquire), "encoding": self.shared.encoding.load(Ordering::Acquire),
            "input_bytes_reserved": self.shared.input.used(), "input_bytes_limit": self.shared.input.limit,
            "memory_bytes_reserved": self.shared.memory.used(), "memory_bytes_limit": self.shared.memory.limit,
            "completed": self.shared.completed.load(Ordering::Acquire), "workers": self.shared.live_workers.load(Ordering::Acquire),
            "inference_elapsed_ms": if inference_started == 0 { 0 } else { tick.saturating_sub(inference_started - 1) },
            "last_progress_age_ms": tick.saturating_sub(self.shared.last_progress.load(Ordering::Acquire)),
            "scheduling": "foreground_3_background_1", "inference_concurrency": 1, "encode_concurrency": 2,
        })
    }
    pub fn shutdown(&self) {
        self.shared.stopping.store(true, Ordering::Release);
        self.shared.admission.close();
        self.shared.changed.notify_all();
        self.shared.memory.changed.notify_all();
    }
    pub async fn join(&self) {
        loop {
            let notified = self.shared.workers_done.notified();
            if self.shared.live_workers.load(Ordering::Acquire) == 0 {
                break;
            }
            notified.await;
        }
        for handle in std::mem::take(&mut *lock(&self.shared.handles)) {
            if handle.join().is_err() {
                tracing::error!("Verarbeitungsworker konnte nicht regulär beendet werden.");
            }
        }
    }
}

struct WorkerExit {
    shared: Arc<Shared>,
    name: &'static str,
}
impl Drop for WorkerExit {
    fn drop(&mut self) {
        if self.name == "backremove-inference" {
            self.shared.engine_alive.store(false, Ordering::Release);
        }
        self.shared.live_workers.fetch_sub(1, Ordering::AcqRel);
        self.shared.workers_done.notify_waiters();
    }
}
fn spawn_worker(
    shared: &Arc<Shared>,
    name: &'static str,
    work: impl FnOnce() + Send + 'static,
) -> Result<()> {
    shared.live_workers.fetch_add(1, Ordering::AcqRel);
    let worker_shared = shared.clone();
    match thread::Builder::new().name(name.into()).spawn(move || {
        let _exit = WorkerExit {
            shared: worker_shared.clone(),
            name,
        };
        if catch_unwind(AssertUnwindSafe(work)).is_err() {
            worker_shared.fail(name);
        }
    }) {
        Ok(handle) => {
            lock(&shared.handles).push(handle);
            Ok(())
        }
        Err(error) => {
            shared.live_workers.fetch_sub(1, Ordering::AcqRel);
            tracing::error!(%error, worker = name, "Workerstart fehlgeschlagen.");
            Err(AppError::Internal)
        }
    }
}

#[derive(Clone, Copy)]
enum Stage {
    Decode,
    Prepared,
    Inference,
    Encode,
}
struct StageLease {
    shared: Arc<Shared>,
    stage: Stage,
}
impl StageLease {
    fn start(shared: &Arc<Shared>, stage: Stage) -> Self {
        let lease = Self {
            shared: shared.clone(),
            stage,
        };
        lease.counter().fetch_add(1, Ordering::AcqRel);
        lease
    }
    fn counter(&self) -> &AtomicUsize {
        match self.stage {
            Stage::Decode => &self.shared.decoding,
            Stage::Prepared => &self.shared.prepared,
            Stage::Inference => &self.shared.inferring,
            Stage::Encode => &self.shared.encoding,
        }
    }
}
impl Drop for StageLease {
    fn drop(&mut self) {
        self.counter().fetch_sub(1, Ordering::AcqRel);
        if matches!(self.stage, Stage::Inference) {
            self.shared.inference_started.store(0, Ordering::Release);
            self.shared.inference_deadline.store(0, Ordering::Release);
        }
        self.shared.progress();
    }
}
struct TokenLease {
    sender: flume::Sender<()>,
}
impl Drop for TokenLease {
    fn drop(&mut self) {
        let _ = self.sender.try_send(());
    }
}
struct PreparedWork {
    job: Job,
    memory: MemoryLease,
    prepared: PreparedImage,
    queued_at: Instant,
    _slot: TokenLease,
    _stage: StageLease,
}
struct EncodeWork {
    job: Job,
    memory: MemoryLease,
    image: DecodedImage,
    mask: Mask,
    geometry: MaskGeometry,
    _slot: TokenLease,
}

fn prepare_loop(
    shared: Arc<Shared>,
    pipeline: Arc<ImagePipeline>,
    sender: flume::Sender<PreparedWork>,
    slots: flume::Receiver<()>,
    return_slot: flume::Sender<()>,
) {
    loop {
        if shared.failed.load(Ordering::Acquire) {
            break;
        }
        // Acquire the single preparation slot BEFORE selecting/decompressing work.
        match slots.recv_timeout(Duration::from_millis(25)) {
            Ok(()) => {}
            Err(flume::RecvTimeoutError::Timeout) => {
                if shared.stopping.load(Ordering::Acquire) && lock(&shared.queue).len() == 0 {
                    break;
                }
                continue;
            }
            Err(flume::RecvTimeoutError::Disconnected) => break,
        }
        let slot = TokenLease {
            sender: return_slot.clone(),
        };
        let Some(mut job) = shared.next_job() else {
            break;
        };
        let inspection = Instant::now();
        job.timings.queue_ms = millis(inspection.saturating_duration_since(job.enqueued));
        let info = match pipeline.inspect(&job.input, &job.content_type) {
            Ok(info) => info,
            Err(error) => {
                job.fail(error);
                continue;
            }
        };
        let inspection_ms = millis(inspection.elapsed());
        let memory_wait = Instant::now();
        let memory = match shared
            .memory
            .acquire(info.estimated_memory_bytes, &job, &shared)
        {
            Ok(memory) => memory,
            Err(error) => {
                job.fail(error);
                continue;
            }
        };
        job.timings.queue_ms += millis(memory_wait.elapsed());
        if let Err(error) = job.check() {
            job.fail(error);
            continue;
        }
        job.control.started.store(true, Ordering::Release);
        let stage = StageLease::start(&shared, Stage::Decode);
        let decode_start = Instant::now();
        let input = std::mem::take(&mut job.input);
        let image = match pipeline.decode(&input, info, job.spec.model) {
            Ok(image) => image,
            Err(error) => {
                job.fail(error);
                continue;
            }
        };
        drop(input);
        drop(job.admission.input.take());
        job.timings.decode_ms = inspection_ms + millis(decode_start.elapsed());
        if let Err(error) = job.check() {
            job.fail(error);
            continue;
        }
        let prepare_start = Instant::now();
        let prepared = match pipeline.prepare(image, &job.spec) {
            Ok(prepared) => prepared,
            Err(error) => {
                job.fail(error);
                continue;
            }
        };
        job.timings.inference_ms = millis(prepare_start.elapsed());
        drop(stage);
        if let Err(error) = job.check() {
            job.fail(error);
            continue;
        }
        let prepared_stage = StageLease::start(&shared, Stage::Prepared);
        let work = PreparedWork {
            job,
            memory,
            prepared,
            queued_at: Instant::now(),
            _slot: slot,
            _stage: prepared_stage,
        };
        if let Err(error) = sender.send(work) {
            error.0.job.fail(AppError::Unavailable);
            break;
        }
    }
}
fn inference_loop(
    shared: Arc<Shared>,
    mut engine: InferenceEngine,
    receiver: flume::Receiver<PreparedWork>,
    sender: flume::Sender<EncodeWork>,
    slots: flume::Receiver<()>,
    return_slot: flume::Sender<()>,
) {
    while let Ok(work) = receiver.recv() {
        let PreparedWork {
            mut job,
            memory,
            prepared,
            queued_at,
            _slot,
            _stage,
        } = work;
        drop(_slot);
        drop(_stage);
        if shared.failed.load(Ordering::Acquire) {
            job.fail(AppError::Unavailable);
            continue;
        }
        if let Err(error) = job.check() {
            job.fail(error);
            continue;
        }
        // Reserve downstream capacity before computing an output. Its slot remains
        // owned until PNG encoding ends, including time queued for an encoder.
        let slot = loop {
            if let Err(error) = job.check() {
                break Err(error);
            }
            match slots.recv_timeout(
                job.deadline
                    .saturating_duration_since(Instant::now())
                    .min(Duration::from_millis(25)),
            ) {
                Ok(()) => {
                    break Ok(TokenLease {
                        sender: return_slot.clone(),
                    });
                }
                Err(flume::RecvTimeoutError::Timeout) => continue,
                Err(flume::RecvTimeoutError::Disconnected) => break Err(AppError::Unavailable),
            }
        };
        let slot = match slot {
            Ok(slot) => slot,
            Err(error) => {
                job.fail(error);
                continue;
            }
        };
        if let Err(error) = job.check() {
            job.fail(error);
            continue;
        }
        job.timings.queue_ms += millis(queued_at.elapsed());
        let PreparedImage {
            image,
            tensor,
            geometry,
        } = prepared;
        shared.inference_started.store(
            shared.born.elapsed().as_millis() as u64 + 1,
            Ordering::Release,
        );
        shared.inference_deadline.store(
            job.deadline
                .saturating_duration_since(shared.born)
                .as_millis() as u64
                + 1,
            Ordering::Release,
        );
        let stage = StageLease::start(&shared, Stage::Inference);
        let inference_start = Instant::now();
        let prediction = engine.infer(job.spec.model, tensor);
        job.timings.inference_ms += millis(inference_start.elapsed());
        // This guard cannot be released by the HTTP future. It ends only AFTER
        // native Session::run returns, also when the waiter cancelled long ago.
        drop(stage);
        let mask = match prediction {
            Ok(mask) => mask,
            Err(error) => {
                job.fail(error);
                shared.fail("inference");
                break;
            }
        };
        if let Err(error) = job.check() {
            job.fail(error);
            continue;
        }
        let work = EncodeWork {
            job,
            memory,
            image,
            mask,
            geometry,
            _slot: slot,
        };
        if let Err(error) = sender.send(work) {
            error.0.job.fail(AppError::Unavailable);
            break;
        }
    }
}
fn encode_loop(
    shared: Arc<Shared>,
    pipeline: Arc<ImagePipeline>,
    receiver: flume::Receiver<EncodeWork>,
) {
    while let Ok(work) = receiver.recv() {
        let EncodeWork {
            mut job,
            mut memory,
            image,
            mask,
            geometry,
            _slot,
        } = work;
        if let Err(error) = job.check() {
            job.fail(error);
            continue;
        }
        let stage = StageLease::start(&shared, Stage::Encode);
        let encoded = match pipeline.encode(image, mask, geometry, &job.spec) {
            Ok(encoded) => encoded,
            Err(error) => {
                job.fail(error);
                continue;
            }
        };
        job.timings.inference_ms += encoded.postprocess_ms;
        job.timings.encode_ms = encoded.encode_ms;
        drop(stage);
        drop(_slot);
        if let Err(error) = job.check() {
            job.fail(error);
            continue;
        }
        if let Err(error) = memory.shrink(encoded.bytes.capacity()) {
            job.fail(error);
            shared.fail("encode-budget");
            break;
        }
        shared.completed.fetch_add(1, Ordering::AcqRel);
        tracing::info!(
            model = job.spec.model.as_str(),
            work_class = job.class.as_str(),
            output_bytes = encoded.bytes.len(),
            admission_ms = job.timings.admission_ms,
            queue_ms = job.timings.queue_ms,
            decode_ms = job.timings.decode_ms,
            inference_ms = job.timings.inference_ms,
            encode_ms = job.timings.encode_ms,
            "Bildverarbeitung abgeschlossen."
        );
        let owner = ResponseOwner {
            bytes: encoded.bytes,
            _memory: memory,
            _admission: job.admission,
        };
        let result = CompletedJob {
            body: Bytes::from_owner(owner),
            model: job.spec.model,
            timings: job.timings,
        };
        let _ = job.reply.send(Ok(result));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn foreground_priority_never_starves_waiting_background() {
        let mut queue = PriorityQueue::default();
        for index in 0..8 {
            queue.push(WorkClass::Foreground, index);
        }
        queue.push(WorkClass::Background, 100);
        queue.push(WorkClass::Background, 101);
        let order: Vec<_> = std::iter::from_fn(|| queue.pop()).collect();
        assert_eq!(order, [0, 1, 2, 100, 3, 4, 5, 101, 6, 7]);
    }

    #[test]
    fn a_lone_class_does_not_accumulate_priority_debt() {
        let mut queue = PriorityQueue::default();
        for index in 0..8 {
            queue.push(WorkClass::Foreground, index);
            assert_eq!(queue.pop(), Some(index));
        }
        queue.push(WorkClass::Background, 100);
        for index in 10..14 {
            queue.push(WorkClass::Foreground, index);
        }
        let order: Vec<_> = std::iter::from_fn(|| queue.pop()).collect();
        assert_eq!(order, [10, 11, 12, 100, 13]);
    }

    #[test]
    fn retained_output_bytes_keep_both_reservations_until_last_drop() {
        let budget = Budget::new(1024);
        let slots = Arc::new(Semaphore::new(1));
        let slot = slots.clone().try_acquire_owned().unwrap();
        let mut memory = budget.try_acquire(800).unwrap();
        let png = vec![1_u8; 32];
        memory.shrink(png.capacity()).unwrap();
        let bytes = Bytes::from_owner(ResponseOwner {
            bytes: png,
            _memory: memory,
            _admission: Admission {
                owner: Weak::new(),
                _slot: slot,
                input: None,
            },
        });
        let retained = bytes.clone();
        drop(bytes);
        assert_eq!(slots.available_permits(), 0);
        assert_eq!(budget.used(), 32);
        drop(retained);
        assert_eq!(slots.available_permits(), 1);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn exceeding_a_budget_never_partially_reserves_it() {
        let budget = Budget::new(100);
        let first = budget.try_acquire(60).unwrap();
        assert!(budget.try_acquire(41).is_none());
        assert_eq!(budget.used(), 60);
        drop(first);
        assert!(budget.try_acquire(100).is_some());
        assert_eq!(budget.used(), 0);
    }
}
