#!/usr/bin/env bash
# Run every model against every registered dataset. Missing corpora are
# reported (status=missing_data in results/<dataset>.csv), not crashed on.
#
#   bash scripts/run_all.sh

set -uo pipefail
cd "$(dirname "$0")/.."

DATASETS=(synthetic_crew_gait osdr_human_spaceflight genelab_omics alsda_physiology osdr_rodent_behaviour inspiration4)
STATUS=0

for dataset in "${DATASETS[@]}"; do
  echo ""
  echo "=================================================================="
  echo "  ${dataset}"
  echo "=================================================================="
  if ! python3 run.py --dataset "${dataset}" --results-csv "results/${dataset}.csv" "$@"; then
    echo "!! ${dataset} had a non-completed cell (see the table above)" >&2
    STATUS=1
  fi
done

echo ""
echo "=================================================================="
echo "  Results"
echo "=================================================================="
ls -1 results/*.csv 2>/dev/null || echo "(none)"

exit "${STATUS}"
