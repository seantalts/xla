Build the `run_hlo_module` XLA tool for CPU. Follow these steps exactly:

## Step 1: Install Bazelisk (if not already installed)
```bash
if ! command -v bazel &>/dev/null; then
  curl -L https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 \
    -o /usr/local/bin/bazel && chmod +x /usr/local/bin/bazel
fi
```

## Step 2: Configure for CPU
```bash
python3 configure.py --backend=CPU
```

## Step 3: Fix Go SDK issue
The proxy blocks go.dev and dl.google.com. Go is pre-installed at /usr/local/go.
Edit `workspace0.bzl` to register the host Go SDK BEFORE the `grpc_extra_deps()` call (~line 153):
- Add to imports: `load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains")`
- Add before `grpc_extra_deps()`: `go_register_toolchains(version = "host")`

## Step 4: Build with proxy workarounds
```bash
export BAZELISK_BASE_URL="https://github.com/bazelbuild/bazel/releases/download"
export GOROOT=/usr/local/go

PROXY_HOST=$(echo "$HTTP_PROXY" | sed -E 's|http://[^@]+@([^:]+):.*|\1|')
PROXY_PORT=$(echo "$HTTP_PROXY" | sed -E 's|.*:([0-9]+)$|\1|')
PROXY_USER=$(echo "$HTTP_PROXY" | sed -E 's|http://([^:]+):.*|\1|')
PROXY_PASS=$(echo "$HTTP_PROXY" | sed -E 's|http://[^:]+:([^@]+)@.*|\1|')

bazel \
  --host_jvm_args="-Djavax.net.ssl.trustStore=/etc/ssl/certs/java/cacerts" \
  --host_jvm_args="-Djavax.net.ssl.trustStorePassword=changeit" \
  --host_jvm_args="-Dhttp.proxyHost=$PROXY_HOST" \
  --host_jvm_args="-Dhttp.proxyPort=$PROXY_PORT" \
  --host_jvm_args="-Dhttps.proxyHost=$PROXY_HOST" \
  --host_jvm_args="-Dhttps.proxyPort=$PROXY_PORT" \
  --host_jvm_args="-Djdk.http.auth.tunneling.disabledSchemes=" \
  --host_jvm_args="-Djdk.http.auth.proxying.disabledSchemes=" \
  --host_jvm_args="-Dhttp.proxyUser=$PROXY_USER" \
  --host_jvm_args="-Dhttp.proxyPassword=$PROXY_PASS" \
  --host_jvm_args="-Dhttps.proxyUser=$PROXY_USER" \
  --host_jvm_args="-Dhttps.proxyPassword=$PROXY_PASS" \
  build //xla/tools:run_hlo_module
```

## Step 5: Verify
```bash
bazel-bin/xla/tools/run_hlo_module --platform=CPU --input_format=hlo xla/tools/data/add.hlo
```

Note: The build may take a long time (XLA is a large C++ project). If additional downloads fail due to proxy restrictions, you may need to pre-download or override those repositories similarly to the Go SDK fix.
