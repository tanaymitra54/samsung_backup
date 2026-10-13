"""Run GSM8K diagnostics; see --help and docs/diagnostic_evaluation.md."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.diagnostic_runner import main
if __name__ == "__main__":
    main("gsm8k")
