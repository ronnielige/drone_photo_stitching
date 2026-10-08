import os, sys, subprocess, time

PY = sys.executable
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(os.path.dirname(SCRIPTS), "cache")

STEPS = [
    ("sfm2.log", "sfm2.py"),
    ("dense_dsm.log", "dense_dsm.py"),
    ("ortho2.log", "ortho2.py"),
    ("verify2.log", "verify2.py"),
]

def wait_for_match2(timeout_s=7200):
    """Wait until matches2.npz exists, is complete (json written before it) and settled."""
    npz = os.path.join(CACHE, "matches2.npz")
    js = os.path.join(CACHE, "matches2.json")
    t0 = time.time()
    print("waiting for match2 outputs ...", flush=True)
    while True:
        if os.path.exists(js) and os.path.exists(npz):
            age = time.time() - os.path.getmtime(npz)
            if age > 60:
                print("match2 outputs ready", flush=True)
                return True
        if time.time() - t0 > timeout_s:
            print("timeout waiting for match2", flush=True)
            return False
        time.sleep(30)

def main():
    if not wait_for_match2():
        sys.exit(1)
    for logname, script in STEPS:
        logf = open(os.path.join(CACHE, logname), "w", encoding="utf-8")
        print("=== running %s ===" % script, flush=True)
        t0 = time.time()
        p = subprocess.run([PY, os.path.join(SCRIPTS, script)],
                          stdout=logf, stderr=subprocess.STDOUT, cwd=SCRIPTS)
        logf.close()
        print("=== %s exit %d (%.0fs) ===" % (script, p.returncode, time.time() - t0), flush=True)
        if p.returncode != 0:
            sys.exit(p.returncode)
    print("ALL DONE", flush=True)

if __name__ == "__main__":
    main()
