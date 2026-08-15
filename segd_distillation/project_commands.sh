#!/usr/bin/env bash
# Entry: setup once, then 4 SEGD settings on GPUs 0–3.
#
#   bash project_commands.sh
#   SKIP_SETUP=1 bash project_commands.sh
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p logs results

if [[ "${SKIP_SETUP:-0}" != "1" ]]; then
  echo "==> running shared setup"
  bash scripts/commands/setup.sh
else
  echo "==> SKIP_SETUP=1 — assuming venv + data already ready"
  # shellcheck disable=SC1091
  source vlm/bin/activate
fi

echo "==> launching slots 1..4"
bash scripts/commands/1.sh > logs/commands_1.log 2>&1 &
pid1=$!
bash scripts/commands/2.sh > logs/commands_2.log 2>&1 &
pid2=$!
bash scripts/commands/3.sh > logs/commands_3.log 2>&1 &
pid3=$!
bash scripts/commands/4.sh > logs/commands_4.log 2>&1 &
pid4=$!

echo "PIDs: 1=$pid1 2=$pid2 3=$pid3 4=$pid4"
echo "Logs: logs/commands_{1,2,3,4}.log"
echo "Waiting..."

fail=0
wait "$pid1" || fail=1
wait "$pid2" || fail=1
wait "$pid3" || fail=1
wait "$pid4" || fail=1

echo "========================================================="
echo "Sweep finished (fail=$fail)"
ls -1 results/*_eval_summary.txt 2>/dev/null || true
echo "========================================================="
exit "$fail"
