#!/usr/bin/env bash
# Unit test for phase3_wake_fleet.sh polling-loop classification.
#
# Reproduces the 2026-05-28 false-positive incident: coordinator unreachable
# for the entire polling window (hub WireGuard down). The script printed
# "Fleet is live" and exited 0 even though /shadow/status never responded.
#
# This test pins three expected exit-code / final-line classifications:
#   T1: coordinator unreachable (closed port)        -> exit 3, "polling failure"
#   T2: coordinator returns malformed JSON           -> exit 3, "polling failure"
#       (JSON parse fails on every tick, so EVER_PARSED_RESPONSE stays 0)
#   T3: coordinator reachable but peers never live   -> exit 2, "timeout"
#
# Run from anywhere; uses absolute paths off this file's location.
# Does not require gh, hcloud, or HCLOUD_TOKEN to actually work -- they are
# stubbed via PATH so the script's existence-checks pass.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$HERE/phase3_wake_fleet.sh"

if [[ ! -x "$SCRIPT" ]]; then
  echo "FAIL: $SCRIPT is not executable" >&2
  exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"; [[ -n "${T3_PID:-}" ]] && kill "$T3_PID" 2>/dev/null || true' EXIT

# ---- PATH stubs ----------------------------------------------------------
STUB_BIN="$TMP/bin"
mkdir -p "$STUB_BIN"

cat >"$STUB_BIN/gh" <<'EOF'
#!/usr/bin/env bash
# Pretend the workflow is already disabled so step 1 is a no-op.
case "$*" in
  "api repos/"*"/actions/workflows/"*" -q .state") echo "disabled_manually" ;;
  *) exit 0 ;;
esac
EOF
chmod +x "$STUB_BIN/gh"

cat >"$STUB_BIN/hcloud" <<'EOF'
#!/usr/bin/env bash
# Report every server as already running so step 2 doesn't try to poweron.
if [[ "$1" == "server" && "$2" == "describe" ]]; then
  echo "running"
fi
exit 0
EOF
chmod +x "$STUB_BIN/hcloud"

export PATH="$STUB_BIN:$PATH"
export HCLOUD_TOKEN="dummy-test-token"

# Force the script's "$BASE/REE_assembly/coordinator.env" to a tmp file we own.
ENV_DIR="$TMP/REE_assembly"
mkdir -p "$ENV_DIR"
ENV_FILE="$ENV_DIR/coordinator.env"
export REE_WORKING="$TMP"

# Fast polling so the test runs in seconds, not minutes.
export TIMEOUT_SECONDS=3
export POLL_INTERVAL=1

PASS=0
FAIL=0

assert_test() {
  local name="$1" expected_rc="$2" expected_substr="$3" actual_rc="$4" actual_out="$5"
  if [[ "$actual_rc" == "$expected_rc" ]] && echo "$actual_out" | grep -q "$expected_substr"; then
    echo "PASS  $name (rc=$actual_rc, matched '$expected_substr')"
    PASS=$((PASS + 1))
  else
    echo "FAIL  $name"
    echo "      expected rc=$expected_rc and final-line substring '$expected_substr'"
    echo "      got rc=$actual_rc"
    echo "      output (last 12 lines):"
    echo "$actual_out" | tail -12 | sed 's/^/        /'
    FAIL=$((FAIL + 1))
  fi
}

# ---- T1: closed-port (coordinator unreachable) ---------------------------
# Port 1 is reserved (tcpmux) and reliably unbound on dev hosts. Use IPv4
# loopback so the connect attempt fails immediately rather than going through
# resolution / firewall hops.
cat >"$ENV_FILE" <<EOF
COORDINATOR_URL=http://127.0.0.1:1
COORDINATOR_LOCAL_TOKEN=dummy-test-token
EOF

set +e
T1_OUT="$(bash "$SCRIPT" --skip-disable-scaler 2>&1)"
T1_RC=$?
set -e

assert_test \
  "T1 closed-port coordinator -> polling-failure classification" \
  3 "exiting on polling failure" "$T1_RC" "$T1_OUT"

# Also verify the script does NOT print the legacy "Fleet is live" false
# positive in this state.
if echo "$T1_OUT" | grep -q "Fleet is live"; then
  echo "FAIL  T1-regression: script printed 'Fleet is live' despite coordinator unreachable"
  FAIL=$((FAIL + 1))
else
  echo "PASS  T1-regression: script did not print 'Fleet is live'"
  PASS=$((PASS + 1))
fi

# ---- T2: malformed JSON --------------------------------------------------
# Start a tiny background HTTP server that returns 200 with non-JSON body.
T2_PORT=18753
"${PYTHON:-/opt/local/bin/python3}" - <<EOF >"$TMP/t2_server.log" 2>&1 &
import http.server, socketserver
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type","text/plain")
        self.end_headers()
        self.wfile.write(b"NOT JSON at all")
    def log_message(self,*a,**k): pass
with socketserver.TCPServer(("127.0.0.1", $T2_PORT), H) as s:
    s.serve_forever()
EOF
T2_PID=$!
# Wait briefly for the server to bind.
for _ in 1 2 3 4 5 6 7 8 9 10; do
  if curl -fsS "http://127.0.0.1:$T2_PORT/shadow/status" >/dev/null 2>&1; then
    break
  fi
  sleep 0.2
done

cat >"$ENV_FILE" <<EOF
COORDINATOR_URL=http://127.0.0.1:$T2_PORT
COORDINATOR_LOCAL_TOKEN=dummy-test-token
EOF

set +e
T2_OUT="$(bash "$SCRIPT" --skip-disable-scaler 2>&1)"
T2_RC=$?
set -e
kill "$T2_PID" 2>/dev/null || true
wait "$T2_PID" 2>/dev/null || true

assert_test \
  "T2 malformed-JSON response -> polling-failure classification" \
  3 "exiting on polling failure" "$T2_RC" "$T2_OUT"

if echo "$T2_OUT" | grep -q "Fleet is live"; then
  echo "FAIL  T2-regression: script printed 'Fleet is live' despite unparseable responses"
  FAIL=$((FAIL + 1))
else
  echo "PASS  T2-regression: script did not print 'Fleet is live'"
  PASS=$((PASS + 1))
fi

# ---- T3: coordinator OK, but peers never reach lifecycle_state=live ------
# Server returns valid JSON with all peers in 'pending'. Should hit timeout
# classification (exit 2), distinct from polling failure.
T3_PORT=18754
"${PYTHON:-/opt/local/bin/python3}" - <<EOF >"$TMP/t3_server.log" 2>&1 &
import http.server, socketserver, json
PAYLOAD = json.dumps({"machines":[
  {"machine":"ree-cloud-1","lifecycle_state":"pending"},
  {"machine":"ree-cloud-2","lifecycle_state":"pending"},
  {"machine":"ree-cloud-3","lifecycle_state":"pending"},
  {"machine":"ree-cloud-4","lifecycle_state":"pending"},
  {"machine":"DLAPTOP-4.local","lifecycle_state":"live"},
]}).encode()
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type","application/json")
        self.end_headers()
        self.wfile.write(PAYLOAD)
    def log_message(self,*a,**k): pass
with socketserver.TCPServer(("127.0.0.1", $T3_PORT), H) as s:
    s.serve_forever()
EOF
T3_PID=$!
for _ in 1 2 3 4 5 6 7 8 9 10; do
  if curl -fsS "http://127.0.0.1:$T3_PORT/shadow/status" >/dev/null 2>&1; then
    break
  fi
  sleep 0.2
done

cat >"$ENV_FILE" <<EOF
COORDINATOR_URL=http://127.0.0.1:$T3_PORT
COORDINATOR_LOCAL_TOKEN=dummy-test-token
EOF

set +e
T3_OUT="$(bash "$SCRIPT" --skip-disable-scaler 2>&1)"
T3_RC=$?
set -e
kill "$T3_PID" 2>/dev/null || true
wait "$T3_PID" 2>/dev/null || true

assert_test \
  "T3 peers-never-live -> timeout classification (exit 2)" \
  2 "exiting on timeout" "$T3_RC" "$T3_OUT"

if echo "$T3_OUT" | grep -q "Fleet is live"; then
  echo "FAIL  T3-regression: script printed 'Fleet is live' despite pending peers"
  FAIL=$((FAIL + 1))
else
  echo "PASS  T3-regression: script did not print 'Fleet is live'"
  PASS=$((PASS + 1))
fi

# ---- Summary -------------------------------------------------------------
echo
echo "=== test_phase3_wake_fleet.sh summary ==="
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
if [[ "$FAIL" -gt 0 ]]; then
  exit 1
fi
exit 0
