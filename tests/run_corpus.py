"""Run the scanner over tests/corpus and compare verdicts with expected.json."""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CORPUS = os.path.join(HERE, "corpus")
VERDICTS = {0: "CLEAN", 1: "SUSPICIOUS", 2: "MALICIOUS", 3: "QUARANTINE"}


def main():
    with open(os.path.join(CORPUS, "expected.json")) as f:
        expected = json.load(f)
    only = sys.argv[1:]
    fails = 0
    for name, want in expected.items():
        if only and name not in only:
            continue
        out_json = os.path.join(CORPUS, name + ".report.json")
        p = subprocess.run([sys.executable, "-m", "pdfscan", os.path.join(CORPUS, name), "--json", out_json],
                           cwd=ROOT, capture_output=True)
        got = VERDICTS.get(p.returncode, f"ERR{p.returncode}")
        ok = got in want
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name:36} got={got:11} want={'/'.join(want)}")
        if not ok:
            print("      " + p.stdout.decode("utf-8", "replace").replace("\n", "\n      ")[:3000])
            if p.stderr:
                print("      STDERR: " + p.stderr.decode("utf-8", "replace")[-1500:])
    print(f"\n{len(expected) - fails if not only else len(only) - fails} passed, {fails} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
