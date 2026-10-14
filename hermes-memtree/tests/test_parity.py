"""Parity guard: the vendored core must equal the originals, plus one recorded patch.

The substrate renders and filters, it never re-implements scoring. Every core file
must be byte-identical to the upstream original it was copied from -- except the
single deviation recorded in patches/core-deviations.patch (the bounded posting
scan), which this test applies to the originals and checks reproduces core exactly.

Run: python3 tests/test_parity.py  (from mcp/)
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VENDORED = os.path.join(ROOT, "memtree_mcp", "core")
UPSTREAM = os.path.join(ROOT, "upstream", "memtree")
PATCH = os.path.join(ROOT, "patches", "core-deviations.patch")
FILES = ["store.py", "watchdog.py", "persist.py", "runtime.py", "geometry.py"]
DEVIATIONS = {"store.py", "watchdog.py"}   # must mirror the patch exactly


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _patch_targets():
    touched = set()
    for line in open(PATCH, encoding="utf-8"):
        if line.startswith("--- ") or line.startswith("+++ "):
            touched.add(os.path.basename(line.split("\t")[0].strip().split("/")[-1]))
    return touched


def main():
    assert os.path.exists(PATCH), f"missing the recorded deviation patch: {PATCH}"
    assert _patch_targets() == DEVIATIONS, \
        f"the patch touches {_patch_targets()}, but DEVIATIONS lists {DEVIATIONS}"

    identical, patched = 0, 0
    missing = [n for n in FILES if not os.path.exists(os.path.join(VENDORED, n))
               or not os.path.exists(os.path.join(UPSTREAM, n))]
    assert not missing, f"missing core/upstream files: {missing}"
    # apply the recorded patch once, for real, to fresh copies of the originals
    work = tempfile.mkdtemp(prefix="memtree-parity-")
    for name in sorted(DEVIATIONS):
        shutil.copy(os.path.join(UPSTREAM, name), os.path.join(work, name))
    p = subprocess.run(["patch", "-p1", "--batch", "--silent", "-i", PATCH],
                       capture_output=True, text=True, cwd=work, timeout=30)
    assert p.returncode == 0, (p.stdout, p.stderr)

    for name in FILES:
        v = os.path.join(VENDORED, name)
        src = os.path.join(UPSTREAM, name)
        if name not in DEVIATIONS:
            assert _sha(v) == _sha(src), f"{name} DRIFTED from the original"
            identical += 1
            print(f"  {name}: identical to the original ({_sha(v)[:12]})")
            continue
        got = _sha(os.path.join(work, name))
        assert got == _sha(v), (
            f"{name}: original + patches/core-deviations.patch != vendored core\n"
            f"  patched={got}\n  vendored={_sha(v)}")
        patched += 1
        print(f"  {name}: original + recorded patch == vendored core ({got[:12]})")

    print(f"\nParity: {identical + patched}/{len(FILES)} core files accounted for "
          f"({identical} byte-identical, {patched} from the recorded patch); "
          f"all scoring math untouched")


if __name__ == "__main__":
    sys.exit(main())
