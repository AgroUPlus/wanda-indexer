#!/usr/bin/env bash
# Runs every offline test. Add --online to also hit the network.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON="${PYTHON:-python3}"
status=0

for suite in test_golden_dsp test_sub_hashes test_pipeline test_progress test_stream_resolver test_db_integration; do
  echo "=== $suite ==="
  if ! "$PYTHON" "tests/$suite.py"; then
    echo "!!! $suite FAILED"
    status=1
  fi
  echo
done

if [[ "${1:-}" == "--online" ]]; then
  echo "=== test_single_track (network) ==="
  "$PYTHON" tests/test_single_track.py || status=1
fi

if [[ $status -eq 0 ]]; then
  echo "All suites passed."
else
  echo "Some suites failed."
fi
exit $status
