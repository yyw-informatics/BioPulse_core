from pathlib import Path
import json
import sys

from biopulse.scorers.dimensionality_reduction_score import score

if __name__ == "__main__":
    benchmark = Path(sys.argv[1])
    run_dir = Path(sys.argv[2])
    result = score(benchmark, run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
