#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Mistral Small 4 prefill at 261,120 tokens, in either topology:
#
#   MODE=1rank   single rank, SP=8 x TP=4 on the whole 8x4 galaxy      ~14.5k tok/s
#   MODE=pp4     PP=4 x [8,1], four column sub-meshes over MeshSockets ~17.2k tok/s   (1.19x)
#
# Mistral-only despite living beside the shared prefill runner: it pins mistral4.json, chunk 5,120
# and the 261,120 headline, and MODE=pp4 needs the [8,1] column bindings, which exist for this
# model alone. One script rather than two so the 1rank/pp4 ratio is measured on identical plumbing.
#
# No pytest for either mode -- Mistral has no demo package, so it runs through prefill_producer
# under ttrun.
set -u

START_TS=$(date +%s)
phase_t=$START_TS
SETUP_S=0; PRODUCER_S=0; DRAIN_S=0
elapsed() { local now; now=$(date +%s); echo $(( now - phase_t )); phase_t=$now; }
hms() { printf '%dm%02ds' $(( $1 / 60 )) $(( $1 % 60 )); }
# Print the wall clock even when we bail out, so a failed run still says how long it burned.
on_exit() {
  local rc=$?
  echo "[repro] ---------------------------------------------------------------"
  echo "[repro] TOTAL WALL $(hms $(( $(date +%s) - START_TS )))  (setup $(hms "$SETUP_S"), producer $(hms "$PRODUCER_S"), drain $(hms "$DRAIN_S"))  exit=$rc"
}
trap on_exit EXIT

MODE="${MODE:?set MODE=1rank or MODE=pp4}"
case "$MODE" in 1rank|pp4) ;; *) echo "[repro] FAIL: MODE must be 1rank or pp4, got '$MODE'" >&2; exit 2 ;; esac

: "${TT_METAL_HOME:?set to your tt-metal checkout, e.g. TT_METAL_HOME=/data/kmabee/tt-metal}"
[ -d "$TT_METAL_HOME/models/demos/common/prefill/runners" ] || {
  echo "[repro] FAIL: TT_METAL_HOME=$TT_METAL_HOME is not a tt-metal checkout" >&2; exit 1; }
export TT_METAL_HOME
T="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$TT_METAL_HOME"
export LD_LIBRARY_PATH="$TT_METAL_HOME/build_Release/lib:${LD_LIBRARY_PATH:-}"
cd "$TT_METAL_HOME"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$TT_METAL_HOME/python_env/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$TT_METAL_HOME/python_env/bin/activate"
fi
python3 -c 'import ttnn' 2>/dev/null || { echo "[repro] FAIL: python3 cannot import ttnn -- build the venv (./create_venv.sh) or activate it"; exit 1; }

TOPO=models/demos/common/prefill/runners/topology_configuration

# MODE selects the rank count and the binding; everything else below is shared.
case "$MODE" in
  1rank) RANKS=1; BASE="$TOPO/pipeline_prefill_request_1rank.yaml" ;;
  # torus_y: its SP-axis RING matches the single-process 8x1 measurements and is what the published
  # headline was measured on -- 298.3 ms/chunk against 338.6 for the plain-2d sibling, so the wrap is
  # worth ~13% and the two are NOT interchangeable for a quoted number.
  pp4)   RANKS=4; BASE="$TOPO/pipeline_prefill_request_intragalaxy_4rank_8x1_torus_y.yaml" ;;
esac

export MISTRAL4_HF_MODEL="${MISTRAL4_HF_MODEL:?set to the Mistral-Small-4-119B checkpoint}"
export PREFILL_HF_MODEL="${PREFILL_HF_MODEL:-$MISTRAL4_HF_MODEL}"
# Resolved as {name}_{arch}_{num_devices}dev/{sp}x{tp}: 1rank sees all 32 devices -> 32dev/8x4,
# each pp4 stage sees 8 -> 8dev/8x1. Different directories, differently-sharded weights, and the
# wrong one does not error -- it silently rebuilds, costing an hour.
export PREFILL_TTNN_CACHE="${PREFILL_TTNN_CACHE:?set to the TTNN weight cache root: 1rank -> the 8x4/32dev one, pp4 -> the 8x1/8dev one}"
# Per-user: a fixed shared path belongs to whoever ran first, and the EACCES surfaces as signal 6
# with the real cause buried in runner.log.
export TT_METAL_CACHE="${TT_METAL_CACHE:-/tmp/tt-metal-cache-${MODE}-$(id -un)}"

export PREFILL_MANIFEST="$TT_METAL_HOME/models/demos/deepseek_v3_d_p/tt/runners/manifests/mistral4.json"
export PREFILL_CHUNK_SIZE=5120        # analyze_prefill_throughput.py assumes this
export PREFILL_NUM_USERS=1            # one long request: 2 slots would double the KV budget here
export PREFILL_KV_ONLY_LAST_LAYER=1   # throughput, not TTFT -- no final norm/LM head, no token
export PREFILL_USE_TRACE=1
export LOGURU_LEVEL=INFO

CHUNKS="${CHUNKS:-51}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-261120}"
# Two requests, and the SECOND is the warm one. Trace capture happens on the first prefill_chunk
# call, so it lands inside request 1; running the script twice does NOT substitute, because each run
# is a fresh process that re-captures. (Running it twice IS still worth it for the JIT kernel cache,
# which is on disk and survives the process -- a different effect. Cold, chunk 0 costs ~13 s.)
REQUESTS="${REQUESTS:-2}"
OUT="${OUT:-$TT_METAL_HOME/mistral4_${MODE}_256k_$(hostname)}"
mkdir -p "$OUT" || { echo "[repro] FAIL: cannot create $OUT" >&2; exit 1; }

# The [8,1] column->device map is PER-GALAXY and a wrong one does NOT error: it builds stages that
# are not columns and reports plausible wrong numbers. Fail closed. An explicit PP_BINDING is
# trusted as-is -- the escape hatch for a binding generated elsewhere or placed by hand.
BASE="${PP_BINDING:-$BASE}"
if [ -z "${PP_BINDING:-}" ] && [ "$MODE" = pp4 ]; then
  HOST_BINDING="${BASE%.yaml}.$(hostname).yaml"
  if [ -f "$HOST_BINDING" ]; then
    BASE="$HOST_BINDING"
  else
    echo "[repro] FAIL: no rank binding for $(hostname) at $HOST_BINDING" >&2
    echo "[repro]       The [8,1] column -> device map is per-galaxy and a wrong one does not error," >&2
    echo "[repro]       it reports plausible wrong numbers. Generate yours first:" >&2
    echo "[repro]         python3 $T/gen_pipeline_binding.py" >&2
    echo "[repro]       Or set PP_BINDING=<file> to choose a binding deliberately." >&2
    exit 1
  fi
fi
[ -f "$BASE" ] || { echo "[repro] FAIL: no rank binding at $BASE" >&2; exit 1; }

# PREFILL_MAX_SEQ_LEN lives in the binding's global_env, which ttrun applies OVER -x, so widening the
# window means rewriting a copy -- exporting it is silently ignored. The stock 1-rank file says 56320,
# and a run that silently keeps it measures 56,320 tokens while claiming to be the 256k cell.
BINDING="$OUT/binding.yaml"
sed "s|PREFILL_MAX_SEQ_LEN: \"[0-9]*\"|PREFILL_MAX_SEQ_LEN: \"$MAX_SEQ_LEN\"|" "$BASE" > "$BINDING"
grep -q "PREFILL_MAX_SEQ_LEN: \"$MAX_SEQ_LEN\"" "$BINDING" || {
  echo "[repro] FAIL: could not set PREFILL_MAX_SEQ_LEN=$MAX_SEQ_LEN in $BINDING" >&2; exit 1; }

DESCRIPTOR=/dev/shm/tt_h2d_stream_service_ds_prefill.bin
rm -f "$DESCRIPTOR"   # a stale one makes the readiness poll pass before this runner is up

echo "[repro] MODE=$MODE ranks=$RANKS chunks=$CHUNKS x $PREFILL_CHUNK_SIZE = $((CHUNKS*PREFILL_CHUNK_SIZE)) tokens, requests=$REQUESTS"
echo "[repro] binding=$BASE"
echo "[repro] cache=$PREFILL_TTNN_CACHE"
echo "[repro] launching $RANKS rank(s) traced ($(date -Is)); logs in $OUT"
setsid python3 ttnn/ttnn/distributed/ttrun.py \
  --rank-binding "$BINDING" \
  --mpi-args "--host $(hostname):$RANKS --map-by slot --bind-to none --tag-output --allow-run-as-root \
              -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x MISTRAL4_HF_MODEL -x PREFILL_HF_MODEL \
              -x PREFILL_MANIFEST -x PREFILL_TTNN_CACHE -x PREFILL_CHUNK_SIZE -x PREFILL_NUM_USERS \
              -x PREFILL_USE_TRACE -x PREFILL_KV_ONLY_LAST_LAYER" \
  -- python3 -m models.demos.common.prefill.runners.prefill_runner \
  > "$OUT/runner.log" 2>&1 &
PGID=$!

# Rank 0 publishes the H2D descriptor once it is serving. No migration here, so that is the only gate.
deadline=$(( $(date +%s) + ${READY_TIMEOUT_S:-3000} ))
while [ ! -e "$DESCRIPTOR" ]; do
  if ! kill -0 "$PGID" 2>/dev/null; then
    echo "[repro] FAIL: runner exited during startup."
    if grep -q 'could not fit in the discovered physical topology' "$OUT/runner.log" 2>/dev/null; then
      if [ "$MODE" = pp4 ]; then
        echo "[repro] The SP axis is not presenting as a torus on this galaxy. Retry with the plain-2d"
        echo "[repro] sibling -- a valid run but NOT comparable to the torus_y headline:"
        echo "[repro]   PP_BINDING=$TOPO/pipeline_prefill_request_intragalaxy_4rank_8x1.yaml MODE=pp4 $0"
      else
        echo "[repro] The 1-rank binding wants a torus on BOTH axes (2d_torus_xy)."
      fi
      echo "[repro] A link that failed to train presents exactly like this; tt-smi -glx_reset has fixed it."
    fi
    echo "[repro] tail:"; tail -40 "$OUT/runner.log"; exit 1
  fi
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "[repro] FAIL: runner not ready within ${READY_TIMEOUT_S:-3000}s; tail:"; tail -40 "$OUT/runner.log"; exit 1
  fi
  sleep 5
done
SETUP_S=$(elapsed)
echo "[repro] runner ready in $(hms "$SETUP_S") ($(date -Is)); starting producer"

PROD_RC=0
PREFILL_MAX_SEQ_LEN=$MAX_SEQ_LEN \
PREFILL_H2D_SERVICE_ID=ds_prefill \
PREFILL_PRODUCER_CHECK_PCC=0 \
PREFILL_PRODUCER_CHUNKS=$CHUNKS \
PREFILL_PRODUCER_MAX_REQUESTS=$REQUESTS \
PREFILL_PRODUCER_INTERLEAVE=round_robin \
PREFILL_SEND_SHUTDOWN=1 \
  timeout "${PRODUCER_TIMEOUT_S:-5400}" python3 -m models.demos.common.prefill.runners.prefill_producer \
  > "$OUT/producer.log" 2>&1 || PROD_RC=$?
PRODUCER_S=$(elapsed)

# The shutdown sentinel drains the pipeline and every rank exits on its own.
for _ in $(seq 1 60); do kill -0 "$PGID" 2>/dev/null || break; sleep 5; done
if kill -0 "$PGID" 2>/dev/null; then
  echo "[repro] runner still up after the sentinel; terminating"
  kill -INT -"$PGID" 2>/dev/null || true; sleep 20; kill -9 -"$PGID" 2>/dev/null || true
fi
DRAIN_S=$(elapsed)

# The producer pushes chunks into a socket and exits 0 whether or not the ranks survived, so its rc
# alone CANNOT see a runner-side crash. Gate on the log too, but on Python-level failures only: rank
# teardown legitimately logs TT_FATAL from the D2D stream-service destructors after device close.
RUNNER_RC=0
if grep -qE 'Traceback \(most recent call last\)|AssertionError' "$OUT/runner.log" 2>/dev/null; then
  RUNNER_RC=3
  echo "[repro] FAIL: rank-level Python failure; this run's numbers are not usable:"
  grep -hoE '(AssertionError|RuntimeError|KeyError|ValueError|TypeError)[^\n]*' "$OUT/runner.log" \
    | sed 's/\x1b\[[0-9;]*m//g' | sort -u | head -5 | sed 's/^/[repro]   /'
fi

# The producer is told how many chunks to send, but the runner's window comes from the binding. If
# those disagree the run completes, reports cleanly, and measures the WRONG ISL -- observed 2026-09-20,
# where a 51-chunk request silently became 11 chunks (56,320 tokens) and nothing flagged it.
SENT=$(grep -oE '\[producer\] .*chunks=\[[0-9,]+\]' "$OUT/producer.log" 2>/dev/null | head -1)
if [ -n "$SENT" ] && ! printf '%s' "$SENT" | grep -q "chunks=\[$CHUNKS"; then
  RUNNER_RC=4
  echo "[repro] FAIL: the producer ran a different chunk count than requested ($CHUNKS)."
  echo "[repro]   $SENT"
  echo "[repro] The measured ISL is not $((CHUNKS*PREFILL_CHUNK_SIZE)) tokens; do not quote this run."
fi

echo "[repro] producer rc=$PROD_RC runner rc=$RUNNER_RC ($(date -Is))"
if [ "$PROD_RC" != 0 ] || [ "$RUNNER_RC" != 0 ]; then exit 1; fi

# Warm-up 8, not the analyzer's default of 4: a multi-chunk interval grows as the KV cache deepens,
# so every published table uses 8 and a 4 does not compare (1rank@102,400 reads 226.7 vs 304.8 ms).
echo "[repro] --- steady-state throughput ---"
python3 "$T/analyze_prefill_throughput.py" "$OUT/runner.log" 8
echo "[repro] --- per-chunk KV-depth ramp ---"
python3 "$T/analyze_prefill_per_chunk.py" "$OUT/runner.log" || true
