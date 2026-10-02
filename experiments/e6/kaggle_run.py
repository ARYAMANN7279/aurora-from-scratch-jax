"""E6 runs on Kaggle's GPUs (handouts/e6_realpde_sim2real.md, amendment A3): one queue per card.

Inputs, two private Kaggle datasets mounted read-only under /kaggle/input:
  ugp2-e6-code: one archive, ugp2_code.tar.gz, holding src/, experiments/e6/, kit/scoring.py (the
                competition kit's, unchanged) and COMMIT; unpacked by Kaggle or, if not, here
  ugp2-e6-data: data/e6 as prep_sim.py wrote it
The repository layout is rebuilt in /tmp/repo (data files linked, run parameters written there). The
lab machine's JAX and optax versions are installed, with its settings: float32 with matmul precision
`highest`, no preallocation. The 15 runs go in two queues, one per card, each in order, so a
fine-tune always follows its own pretrain. With one card, the queues run one after the other.
Output in /kaggle/working: rows.jsonl (rows_q0 and rows_q1 merged), logs/, runs/ (parameters),
env.json.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time

OUT = "/kaggle/working"
W = "/tmp/repo"
PINS = ["jax[cuda12]==0.10.2", "optax==0.2.8", "numpy==2.4.6", "ml_dtypes==0.6.0", "scipy==1.17.1"]
QUEUES = [
    [("S", None, 0), ("PD", "pretrain", 0), ("PD", "finetune", 0), ("PA", "pretrain", 0), ("PA", "finetune", 0),
     ("S", None, 2), ("PD", "pretrain", 2), ("PD", "finetune", 2)],
    [("S", None, 1), ("PD", "pretrain", 1), ("PD", "finetune", 1), ("PA", "pretrain", 1), ("PA", "finetune", 1),
     ("PA", "pretrain", 2), ("PA", "finetune", 2)],
]


def sh(cmd):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if len(hits) != 1:
        raise SystemExit(f"expected one {name} under /kaggle/input, found {hits}")
    return hits[0]


def code_root():
    hits = glob.glob("/kaggle/input/**/experiments/e6/run_e6.py", recursive=True)
    if not hits:
        shutil.unpack_archive(find("ugp2_code.tar.gz"), "/tmp/code")
        hits = glob.glob("/tmp/code/**/experiments/e6/run_e6.py", recursive=True)
    if len(hits) != 1:
        raise SystemExit(f"expected one experiments/e6/run_e6.py, found {hits}")
    return hits[0][: -len("experiments/e6/run_e6.py")]


code = code_root()
data = os.path.dirname(find("sim_frames.npy"))
for part in ("src", "experiments"):
    shutil.copytree(os.path.join(code, part), os.path.join(W, part), dirs_exist_ok=True)
os.makedirs(f"{W}/data/e6", exist_ok=True)
for f in os.listdir(data):
    if not os.path.exists(f"{W}/data/e6/{f}"):
        os.symlink(os.path.join(data, f), f"{W}/data/e6/{f}")
commit = open(os.path.join(code, "COMMIT")).read().strip()
os.makedirs(f"{OUT}/logs", exist_ok=True)

sh(f"{sys.executable} -m pip install -q " + " ".join(f"'{p}'" for p in PINS))
gpus = [g for g in sh("nvidia-smi -L").splitlines() if g.startswith("GPU")]
env = dict(os.environ, JAX_DEFAULT_MATMUL_PRECISION="highest", XLA_PYTHON_CLIENT_PREALLOCATE="false",
           CUDA_DEVICE_ORDER="PCI_BUS_ID", E6_KIT=os.path.join(code, "kit", "scoring.py"))
info = dict(commit=commit, gpus=gpus, python=sys.version, pins=PINS,
            versions=sh(f"{sys.executable} -m pip freeze").splitlines(), started=time.strftime("%FT%T"))
json.dump(info, open(f"{OUT}/env.json", "w"), indent=1)
print(json.dumps({k: info[k] for k in ("commit", "gpus", "python")}, indent=1), flush=True)

smoke = subprocess.run([sys.executable, "experiments/e6/run_e6.py", "--arm", "S", "--seed", "0", "--smoke", "--steps", "100"],
                       cwd=W, env=dict(env, CUDA_VISIBLE_DEVICES="0"), capture_output=True, text=True)
open(f"{OUT}/logs/smoke.log", "w").write(smoke.stdout + smoke.stderr)
if smoke.returncode:
    raise SystemExit(f"smoke test failed (rc={smoke.returncode}); see logs/smoke.log")
print("smoke test passed", flush=True)


def run_queue(q, gpu):
    for arm, stage, seed in q:
        cmd = [sys.executable, "experiments/e6/run_e6.py", "--arm", arm, "--seed", str(seed),
               "--results", f"{OUT}/rows_q{gpu}.jsonl"] + (["--stage", stage] if stage else [])
        t0 = time.time()
        with open(f"{OUT}/logs/q{gpu}.log", "a") as f:
            f.write(f"== {' '.join(cmd[1:])}\n")
            f.flush()
            rc = subprocess.run(cmd, cwd=W, env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=f,
                                stderr=subprocess.STDOUT).returncode
            f.write(f"== rc={rc} after {time.time() - t0:.0f} s\n")
        print(f"card {gpu}: {arm} {stage or ''} seed {seed}: rc={rc}, {time.time() - t0:.0f} s", flush=True)


if len(gpus) >= 2:
    threads = [threading.Thread(target=run_queue, args=(q, g)) for g, q in enumerate(QUEUES)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
else:
    for q in QUEUES:
        run_queue(q, 0)

rows = []
for p in sorted(glob.glob(f"{OUT}/rows_q*.jsonl")):
    for line in open(p):
        r = json.loads(line)
        r["git_commit"] = r.get("git_commit") or commit
        rows.append(r)
with open(f"{OUT}/rows.jsonl", "w") as f:
    for r in sorted(rows, key=lambda r: (r["seed"], r["name"])):
        f.write(json.dumps(r) + "\n")
if os.path.isdir(f"{W}/data/e6/runs"):
    shutil.copytree(f"{W}/data/e6/runs", f"{OUT}/runs", dirs_exist_ok=True)
print(f"done: {len(rows)} rows of 15", flush=True)
