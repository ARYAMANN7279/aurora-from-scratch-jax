"""Stage 4 F3 on Kaggle's GPUs (handouts/stage04_perceiver.md, section 5, F3): the shared model trained longer.

Input: one private Kaggle dataset, ugp2-s4f3, with two archives (unpacked by Kaggle or, if not, here):
  ugp2_s4f3_code.tar.gz: src/, experiments/stage4/, COMMIT
  ugp2_s4f3_data.tar:    data/stage2/acc_v1/re300_kf4/seed_000{0..6}/frames_{64,128}.npy (the 16², 64² and 128²
                         frames) and data/stage4/f512/re1000_kf4/seed_0000/frames_512.npy (the 512² frames)
The repository layout is rebuilt in /tmp/repo, with data/ linked to the dataset. JAX and optax are pinned to the lab
machine's versions, float32 with matmul precision `highest`, JAX_PLATFORMS=cuda; the job stops if JAX sees no GPU.
MODE "time" times 40 steps (10 per resolution) on card 0 and writes nothing else. MODE "run" trains the shared model
for STEPS steps at each seed in SEEDS, one seed per card at the same time.
Output in /kaggle/working: f3_main_*.json (train_perceiver.py's results), logs/, env.json.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time

MODE = "run"
STEPS = 96000
SEEDS = (0, 1)
OUT = "/kaggle/working"
W = "/tmp/repo"
PINS = ["jax[cuda12]==0.10.2", "optax==0.2.8", "numpy==2.4.6", "ml_dtypes==0.6.0", "scipy==1.17.1"]


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


code = root_of("experiments/stage4/train_perceiver.py", "ugp2_s4f3_code.tar.gz", "/tmp/code")
data = root_of("data/stage4/f512/re1000_kf4/seed_0000/frames_512.npy", "ugp2_s4f3_data.tar", "/tmp/data")
for part in ("src", "experiments"):
    shutil.copytree(os.path.join(code, part), os.path.join(W, part), dirs_exist_ok=True)
os.symlink(os.path.join(data, "data"), f"{W}/data")
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
info = dict(commit=commit, mode=MODE, steps=STEPS, seeds=SEEDS, gpus=gpus, jax_device=jax_dev, python=sys.version,
            pins=PINS, versions=sh(f"{sys.executable} -m pip freeze").splitlines(), started=time.strftime("%FT%T"))
json.dump(info, open(f"{OUT}/env.json", "w"), indent=1)
print(json.dumps({k: info[k] for k in ("commit", "mode", "gpus", "jax_device")}, indent=1), flush=True)


def train(args, gpu, log):
    t0 = time.time()
    with open(log, "a") as f:
        rc = subprocess.run([sys.executable, "experiments/stage4/train_perceiver.py", *args], cwd=W,
                            env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=f, stderr=subprocess.STDOUT).returncode
        f.write(f"== rc={rc} after {time.time() - t0:.0f} s\n")
    print(f"card {gpu}: {' '.join(args)}: rc={rc}, {time.time() - t0:.0f} s", flush=True)


if MODE == "time":  # card 0 times; card 1 makes a 500-step smoke run through evaluation and the results file
    jobs = [(["--run", "main", "--steps", "40", "--time-only"], 0, f"{OUT}/logs/time.log"),
            (["--run", "main", "--steps", "500", "--seed", "1", "--out", "smoke_f3"], 1, f"{OUT}/logs/smoke.log")]
    threads = [threading.Thread(target=train, args=j) for j in jobs[: len(gpus)]]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for j in jobs[: len(gpus)]:
        print(open(j[2]).read(), flush=True)
    sys.exit()

runs = [(["--run", "main", "--steps", str(STEPS), "--seed", str(s), "--out", f"f3_main_{STEPS // 1000}k_s{s}"], g)
        for g, s in enumerate(SEEDS)]
if len(gpus) < len(runs):
    raise SystemExit(f"F3 runs one seed per card and needs {len(runs)} cards: {gpus}")
threads = [threading.Thread(target=train, args=(a, g, f"{OUT}/logs/s{SEEDS[g]}.log")) for a, g in runs]
for t in threads:
    t.start()
for t in threads:
    t.join()
for p in glob.glob(f"{W}/experiments/stage4/results/f3_main_*.json"):
    r = json.load(open(p))
    r["git_commit"] = commit
    json.dump(r, open(os.path.join(OUT, os.path.basename(p)), "w"), indent=1)
    print(os.path.basename(p), r["eval_rel_err"], flush=True)
