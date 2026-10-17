"""Run one command on an idle card 1–3, checking placement, heat and other users.

Usage: glane3.sh -- <command> [args ...]. Only this launcher's process group is signalled.
Paused jobs release their context after 60 seconds and retry (at most three retries).
Scratch and caches stay under the project. Card 0 is always excluded.
"""
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def query(kind, fields):
    out = subprocess.check_output(["nvidia-smi", f"--query-{kind}={fields}",
                                   "--format=csv,noheader,nounits"], text=True)
    return [[x.strip() for x in row.split(",")] for row in out.splitlines() if row.strip()]


def snapshot():
    cards = query("gpu", "index,uuid,temperature.gpu,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown")
    apps = query("compute-apps", "gpu_uuid,pid")
    return cards, [(uuid, int(pid)) for uuid, pid in apps]


def throttled(row):
    return any(x.lower() not in ("not active", "n/a", "[n/a]", "not supported", "[not supported]") for x in row[3:])


def in_group(pid, pgid):
    try:
        return os.getpgid(pid) == pgid
    except ProcessLookupError:
        return False


def stop(child):
    if child.poll() is not None:
        return
    for sig in (signal.SIGCONT, signal.SIGTERM):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            return
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def main(command):
    scratch = ROOT / "data/feedback_scratch"
    for part in ("tmp", "cache", "mpl", "claims"):
        (scratch / part).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", JAX_PLATFORMS="cuda",
               JAX_DEFAULT_MATMUL_PRECISION="highest", XLA_PYTHON_CLIENT_PREALLOCATE="false",
               TMPDIR=str(scratch / "tmp"), MPLCONFIGDIR=str(scratch / "mpl"),
               XDG_CACHE_HOME=str(scratch / "cache"), JAX_COMPILATION_CACHE_DIR=str(scratch / "cache/jax"),
               JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="1")
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("BASH_ENV", None)
    child = None

    def interrupted(signum, _):
        if child is not None:
            stop(child)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    for attempt in range(4):
        claim = None
        while claim is None:
            cards, apps = snapshot()
            for row in cards:
                idx, uuid, temp = row[:3]
                if idx not in ("1", "2", "3") or int(temp) >= 85 or throttled(row):
                    continue
                lock = open(scratch / "claims" / f"gpu{idx}.lock", "a")
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lock.close()
                    continue
                cards2, apps2 = snapshot()
                now = next(r for r in cards2 if r[1] == uuid)
                if any(u == uuid for u, _ in apps2) or int(now[2]) >= 85 or throttled(now):
                    lock.close()
                    continue
                claim = lock
                break
            if claim is None:
                time.sleep(10)
        env["CUDA_VISIBLE_DEVICES"] = uuid
        print(f"[guard] attempt={attempt + 1} card={idx} uuid={uuid} command={command}", flush=True)
        paused = None
        try:
            child = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
            while child.poll() is None:
                cards, apps = snapshot()
                row = next(r for r in cards if r[1] == uuid)
                ours = [(u, p) for u, p in apps if in_group(p, child.pid)]
                if any(u != uuid for u, _ in ours):
                    print("[guard] placement interlock: stopping own job", flush=True)
                    stop(child)
                    return 98
                foreign = any(u == uuid and not in_group(p, child.pid) for u, p in apps)
                hot = int(row[2]) >= 87 or throttled(row)
                if paused is None and (hot or foreign):
                    os.killpg(child.pid, signal.SIGSTOP)
                    paused = time.monotonic()
                    print(f"[guard] paused temp={row[2]} foreign={foreign}", flush=True)
                elif paused is not None:
                    if not foreign and int(row[2]) <= 80 and not throttled(row):
                        os.killpg(child.pid, signal.SIGCONT)
                        paused = None
                        print("[guard] resumed", flush=True)
                    elif time.monotonic() - paused >= 60:
                        print("[guard] releasing paused context and requeuing", flush=True)
                        stop(child)
                        break
                time.sleep(2)
            if child.returncode == 0:
                return 0
            print(f"[guard] own job exit={child.returncode}", flush=True)
        finally:
            if child is not None:
                stop(child)
            claim.close()
    return 1


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--"]:
        args = args[1:]
    if not args:
        raise SystemExit(__doc__)
    raise SystemExit(main(args))
