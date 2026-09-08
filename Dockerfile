# syntax=docker/dockerfile:1
# All three official image manifests are pinned; target is deliberately linux/amd64.
FROM python:3.12.9-slim-bookworm@sha256:48a11b7ba705fd53bf15248d1f94d36c39549903c5d59edcfa2f3f84126e7b44 AS artifacts
WORKDIR /src
COPY tools/requirements-prepare.txt ./tools/
RUN python -m pip install --disable-pip-version-check --timeout 30 --retries 2 \
    --require-hashes -r tools/requirements-prepare.txt
COPY tools/ ./tools/
RUN --mount=type=cache,target=/src/.artifacts-cache,sharing=locked \
    python -B tools/prepare.py --target linux-cpu --output /artifact-pack

FROM rust:1.98.0-bookworm@sha256:82150a52ec202c1b14d7817e14516c392bb7f5cfebd88f1ed531cb37ebd39922 AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
    meson ninja-build nasm pkg-config python3 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /tmp/dav1d \
    && curl --fail --location --proto '=https' --tlsv1.2 --max-time 300 --retry 2 \
       https://github.com/videolan/dav1d/archive/refs/tags/1.5.3.tar.gz -o /tmp/dav1d.tar.gz \
    && echo '8d976b93135213d41385c20205475269a6826a68ebfd716c4d9a7a3ff2a79703e8df0573e43207c81b5db44807d2721db18ec84c0fc6bef98efab86a2cccb6cc  /tmp/dav1d.tar.gz' | sha512sum --check --strict \
    && tar -xzf /tmp/dav1d.tar.gz --strip-components=1 -C /tmp/dav1d \
    && meson setup /tmp/dav1d/build /tmp/dav1d --prefix=/opt/dav1d --libdir=lib \
       --buildtype=release --default-library=shared -Denable_tools=false -Denable_tests=false \
    && meson compile -C /tmp/dav1d/build \
    && meson install -C /tmp/dav1d/build
ENV PKG_CONFIG_PATH=/opt/dav1d/lib/pkgconfig \
    SYSTEM_DEPS_DAV1D_BUILD_INTERNAL=never
ENV LD_LIBRARY_PATH=/opt/dav1d/lib
WORKDIR /src
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY src/ ./src/
COPY vendor/ ./vendor/
COPY tools/ ./tools/
COPY Dockerfile ./
COPY --from=artifacts /artifact-pack ./artifacts/linux-cpu/
ARG SOURCE_REVISION=local
ENV SOURCE_REVISION=${SOURCE_REVISION}
RUN cargo fetch --locked \
    && cargo fmt --all -- --check \
    && cargo clippy --all-targets --release --locked --offline -- -D warnings \
    && cargo test --release --locked --offline \
    && cargo build --release --locked --offline \
    && mkdir -p /out/licenses/dav1d /out/provenance \
    && cp target/release/backremove /out/backremove \
    && cp -a artifacts /out/artifacts \
    && cp /tmp/dav1d/COPYING /out/licenses/dav1d/COPYING.txt \
    && dpkg-query -W > /out/provenance/build-system-packages.txt \
    && python3 -B tools/build_inventory.py --output /out --dav1d-prefix /opt/dav1d

# Native executable, native libraries, models, fonts and licenses only.
# No Python executable/modules, PyTorch, exporter, compiler or CUDA toolkit.
FROM debian:bookworm-slim@sha256:88200866dfff7ea7f5cbcb6ec7c8a701889efe6fe859fe64d6990e4b07ea4171 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --user-group --no-create-home --shell /usr/sbin/nologin backremove
COPY --from=build /opt/dav1d/lib/libdav1d.so.7.0.0 /usr/local/lib/
COPY --from=build /out/ /opt/backremove/
RUN ldconfig \
    && dpkg-query -W > /opt/backremove/provenance/runtime-system-packages.txt \
    && chmod -R a-w /opt/backremove
WORKDIR /opt/backremove
ENV HOST=0.0.0.0 PORT=8000 INFERENCE_DEVICE=cpu QUALITY_MODEL_ENABLED=0 \
    ARTIFACT_MANIFEST=/opt/backremove/artifacts/linux-cpu/manifest.json \
    FAST_TIMEOUT=9 QUALITY_TIMEOUT=29 SHUTDOWN_GRACE=30
ARG SOURCE_REVISION=local
LABEL org.opencontainers.image.title="BackRemove native CPU" \
      org.opencontainers.image.description="Native Rust background removal. Built with DINOv3." \
      org.opencontainers.image.source="https://github.com/sellaro-net/backremove" \
      org.opencontainers.image.version="2.0.0" \
      org.opencontainers.image.revision="${SOURCE_REVISION}"
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["/opt/backremove/backremove", "--healthcheck"]
STOPSIGNAL SIGTERM
ENTRYPOINT ["/opt/backremove/backremove"]
