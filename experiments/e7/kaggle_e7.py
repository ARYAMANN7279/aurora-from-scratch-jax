"""E7 on Kaggle's GPUs (handouts/e7_attention_rpb.md, section 4): arms B and R, one queue per card.

Input: one private Kaggle dataset, ugp2-e7, with two archives (unpacked by Kaggle or, if not, here):
  ugp2_e7_code.tar.gz: src/, experiments/run.py, experiments/e7/, COMMIT
  ugp2_e7_data.tar:    data/stage2/e1_v1/re300_kf4/seed_000{0..5}/frames_64.npy (E1's scratch task) and
                       the dataset's manifest.json and dataset.json (the harness checks splits against the
                       manifest; version 1 of this job stopped at its smoke run without it)
The repository layout is rebuilt in /tmp/repo, with data/ a writable folder (the harness saves parameters to
data/runs/) whose stage2/ links to the dataset. JAX and optax are pinned to the lab machine's versions, float32
with matmul precision `highest`. Every run sets JAX_PLATFORMS=cuda, and the job stops before any run if JAX
sees no GPU or a 20-step smoke run of B or R fails. The queues alternate the arms across the cards: card 0
runs B0, R1, B2 and card 1 runs R0, B1, R2.
Output in /kaggle/working: rows.jsonl, logs/, env.json.
"""
import copy
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
QUEUES = [["e7_B_s0", "e7_R_s1", "e7_B_s2"], ["e7_R_s0", "e7_B_s1", "e7_R_s2"]]


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


def root_of(marker, archive, dest):
    """The folder holding `marker` (a relative path), unpacking `archive` into dest if Kaggle did not."""
    hits = glob.glob(f"/kaggle/input/**/{marker}", recursive=True)
    if not hits:
        shutil.unpack_archive(find(archive), dest)
        hits = glob.glob(f"{dest}/**/{marker}", recursive=True)
    if len(hits) != 1:
        raise SystemExit(f"expected one {marker}, found {hits}")
    return hits[0][: -len(marker)]


code = root_of("experiments/e7/configs.json", "ugp2_e7_code.tar.gz", "/tmp/code")
data = root_of("data/stage2/e1_v1/re300_kf4/seed_0000/frames_64.npy", "ugp2_e7_data.tar", "/tmp/data")
for part in ("src", "experiments"):
    shutil.copytree(os.path.join(code, part), os.path.join(W, part), dirs_exist_ok=True)
os.makedirs(f"{W}/data", exist_ok=True)
os.symlink(os.path.join(data, "data", "stage2"), f"{W}/data/stage2")
commit = open(os.path.join(code, "COMMIT")).read().strip()
os.makedirs(f"{OUT}/logs", exist_ok=True)

sh(f"{sys.executable} -m pip install -q " + " ".join(f"'{p}'" for p in PINS))
probe = "import jax; d = jax.devices()[0]; print(d.platform, d.device_kind)"
jax_dev = sh(f"JAX_PLATFORMS=cuda {sys.executable} -c '{probe}'").strip()
if not jax_dev.startswith(("gpu", "cuda")):
    raise SystemExit(f"JAX does not see a GPU: {jax_dev}")
gpus = [g for g in sh("nvidia-smi -L").splitlines() if g.startswith("GPU")]
env = dict(os.environ, JAX_DEFAULT_MATMUL_PRECISION="highest", XLA_PYTHON_CLIENT_PREALLOCATE="false",
           CUDA_DEVICE_ORDER="PCI_BUS_ID", JAX_PLATFORMS="cuda")
info = dict(commit=commit, gpus=gpus, jax_device=jax_dev, python=sys.version, pins=PINS,
            versions=sh(f"{sys.executable} -m pip freeze").splitlines(), started=time.strftime("%FT%T"))
json.dump(info, open(f"{OUT}/env.json", "w"), indent=1)
print(json.dumps({k: info[k] for k in ("commit", "gpus", "jax_device")}, indent=1), flush=True)
configs = {c["name"]: c for c in json.load(open(f"{W}/experiments/e7/configs.json"))}


def run(cfg, gpu, results, log):
    path = f"{W}/experiments/e7/run_{cfg['name']}.json"
    json.dump(cfg, open(path, "w"))
    t0 = time.time()
    with open(log, "a") as f:
        f.write(f"== {cfg['name']}\n")
        f.flush()
        rc = subprocess.run([sys.executable, "experiments/run.py", path, "--results", results], cwd=W,
                            env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=f, stderr=subprocess.STDOUT).returncode
        f.write(f"== rc={rc} after {time.time() - t0:.0f} s\n")
    print(f"card {gpu}: {cfg['name']}: rc={rc}, {time.time() - t0:.0f} s", flush=True)
    return rc


for name in ("e7_B_s0", "e7_R_s0"):  # 20 steps each, through the whole harness path; no rows kept
    smoke = copy.deepcopy(configs[name])
    smoke["name"], smoke["train"]["steps"] = "smoke_" + name, 20
    if run(smoke, 0, "/tmp/smoke_rows.jsonl", f"{OUT}/logs/smoke.log"):
        raise SystemExit(f"smoke run {name} failed; see logs/smoke.log")
print("smoke test passed", flush=True)


def run_queue(q, gpu):
    for name in q:
        run(configs[name], gpu, f"{OUT}/rows_q{gpu}.jsonl", f"{OUT}/logs/q{gpu}.log")


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
    for r in sorted(rows, key=lambda r: r["name"]):
        f.write(json.dumps(r) + "\n")
print(f"done: {len(rows)} rows of 6", flush=True)
