"""Reproduce everything from the raw files in data/raw.

    python run_all.py                   # full pipeline (about 20 min on 2 cores; most of it is SARIMAX fits)
    python run_all.py --skip-forecast   # reuse existing forecast outputs

Steps: tests -> clean + aggregate (DuckDB) -> EDA -> forecasting -> inference -> finding windows
-> notebook -> resume facts.
"""
import subprocess
import sys
import time

STEPS = [
    ("tests", [sys.executable, "-m", "pytest", "-q", "tests"]),
    ("clean + aggregate", [sys.executable, "-m", "src.pipeline"]),
    ("eda", [sys.executable, "-m", "src.eda"]),
    ("forecast", [sys.executable, "-m", "src.forecast"]),
    ("inference", [sys.executable, "-m", "src.inference"]),
    ("finding windows", [sys.executable, "-m", "src.findings_stats"]),
    ("notebook", [sys.executable, "-m", "src.build_notebook"]),
    ("resume facts", [sys.executable, "-m", "src.resume_facts"]),
]


def main() -> None:
    skip = {"forecast"} if "--skip-forecast" in sys.argv else set()
    for name, cmd in STEPS:
        if name in skip:
            print(f"== {name}: skipped")
            continue
        t = time.time()
        print(f"== {name}", flush=True)
        subprocess.run(cmd, check=True)
        print(f"== {name} done in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
