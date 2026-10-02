"""E6 follow-up F1 on Kaggle's GPUs (handouts/e6_realpde_sim2real.md, section 10): the PD and PA pretrained
models trained on real data with S's exact recipe, one queue per card.

Inputs, mounted read-only under /kaggle/input:
  ugp2-e6-code: the code archive (ugp2_code.tar.gz: src/, experiments/e6/, kit/scoring.py, COMMIT)
  ugp2-e6-data: data/e6 as prep_sim.py wrote it
  the output of the job ugp2-e6-runs: runs/e6_{PD,PA}_pretrain_s{0,1,2}/params.npz, the pretrained
  models whose rows E6 scored
As in kaggle_run.py: the repository layout is rebuilt in /tmp/repo, JAX and optax are pinned to the lab
machine's versions, float32 with matmul precision `highest`. Every run sets JAX_PLATFORMS=cuda, and the
job stops before any run if JAX sees no GPU or a pretrained model is missing.
Output in /kaggle/working: rows_f1.jsonl, logs/, runs/ (the new parameters), env.json (with the sha256
of every pretrained model used).
"""
import glob
import hashlib
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
QUEUES = [[("PD", s) for s in (0, 1, 2)], [("PA", s) for s in (0, 1, 2)]]
PRETRAINED = [f"e6_{a}_pretrain_s{s}" for a in ("PD", "PA") for s in (0, 1, 2)]


def sh(cmd):
    print("+", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
        raise SystemExit(f"failed (rc={r.returncode}): {cmd}")
    return r.stdout


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
os.makedirs(f"{W}/data/e6/runs", exist_ok=True)
for f in os.listdir(data):
    if not os.path.exists(f"{W}/data/e6/{f}"):
        os.symlink(os.path.join(data, f), f"{W}/data/e6/{f}")
sha = {}
for name in PRETRAINED:
    src = find(f"runs/{name}/params.npz")
    os.makedirs(f"{W}/data/e6/runs/{name}", exist_ok=True)
    os.symlink(src, f"{W}/data/e6/runs/{name}/params.npz")
    sha[name] = hashlib.sha256(open(src, "rb").read()).hexdigest()
commit = open(os.path.join(code, "COMMIT")).read().strip()
os.makedirs(f"{OUT}/logs", exist_ok=True)

sh(f"{sys.executable} -m pip install -q " + " ".join(f"'{p}'" for p in PINS))
jax_dev = sh(f"JAX_PLATFORMS=cuda {sys.executable} -c 'import jax; d = jax.devices()[0]; print(d.platform, d.device_kind)'").strip()
if not jax_dev.startswith(("gpu", "cuda")):
    raise SystemExit(f"JAX does not see a GPU: {jax_dev}")
gpus = [g for g in sh("nvidia-smi -L").splitlines() if g.startswith("GPU")]
env = dict(os.environ, JAX_DEFAULT_MATMUL_PRECISION="highest", XLA_PYTHON_CLIENT_PREALLOCATE="false",
           CUDA_DEVICE_ORDER="PCI_BUS_ID", JAX_PLATFORMS="cuda", E6_KIT=os.path.join(code, "kit", "scoring.py"))
info = dict(commit=commit, gpus=gpus, jax_device=jax_dev, python=sys.version, pins=PINS, pretrained_sha256=sha,
            versions=sh(f"{sys.executable} -m pip freeze").splitlines(), started=time.strftime("%FT%T"))
json.dump(info, open(f"{OUT}/env.json", "w"), indent=1)
print(json.dumps({k: info[k] for k in ("commit", "gpus", "jax_device")}, indent=1), flush=True)

smoke = subprocess.run([sys.executable, "experiments/e6/run_e6.py", "--arm", "S", "--seed", "0", "--smoke", "--steps", "100"],
                       cwd=W, env=dict(env, CUDA_VISIBLE_DEVICES="0"), capture_output=True, text=True)
open(f"{OUT}/logs/smoke.log", "w").write(smoke.stdout + smoke.stderr)
if smoke.returncode:
    raise SystemExit(f"smoke test failed (rc={smoke.returncode}); see logs/smoke.log")
print("smoke test passed", flush=True)


def run_queue(q, gpu):
    for arm, seed in q:
        cmd = [sys.executable, "experiments/e6/run_e6.py", "--arm", arm, "--stage", "finetune_matched", "--seed",
               str(seed), "--results", f"{OUT}/rows_f1_q{gpu}.jsonl"]
        t0 = time.time()
        with open(f"{OUT}/logs/q{gpu}.log", "a") as f:
            f.write(f"== {' '.join(cmd[1:])}\n")
            f.flush()
            rc = subprocess.run(cmd, cwd=W, env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=f,
                                stderr=subprocess.STDOUT).returncode
            f.write(f"== rc={rc} after {time.time() - t0:.0f} s\n")
        print(f"card {gpu}: {arm} finetune_matched seed {seed}: rc={rc}, {time.time() - t0:.0f} s", flush=True)


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
for p in sorted(glob.glob(f"{OUT}/rows_f1_q*.jsonl")):
    for line in open(p):
        r = json.loads(line)
        r["git_commit"] = r.get("git_commit") or commit
        rows.append(r)
with open(f"{OUT}/rows_f1.jsonl", "w") as f:
    for r in sorted(rows, key=lambda r: (r["seed"], r["name"])):
        f.write(json.dumps(r) + "\n")
for name in glob.glob(f"{W}/data/e6/runs/e6_*_finetune_matched_s*"):
    shutil.copytree(name, f"{OUT}/runs/{os.path.basename(name)}", dirs_exist_ok=True)
print(f"done: {len(rows)} rows of 6", flush=True)
