"""Stage 12 supplement on Kaggle's T4s (handouts/stage12_speed.md, section 9): every arm in one job.

Input: one private Kaggle dataset, ugp2-s12, with two archives (unpacked by Kaggle or, if not, here):
  ugp2_s12_code.tar.gz: src/, the scripts in experiments/stage12/, experiments/e1/configs/pretrain.json,
                        COMMIT
  ugp2_s12_data.tar:    data/stage12/ref_e1_cpu.npz, data/runs/2f09979ae05dea9b/params.npz and the four
                        re300_kf4 test trajectories (data/stage2/e1_v1/...) for the accuracy check
The repository layout is rebuilt in /tmp/repo. As on the lab machine, JAX and PyTorch get separate
environments, pinned to the lab machine's versions: JAX in the image's own Python (a fresh venv's JAX fell
back to the CPU on 2 Oct, version 1 of this job), PyTorch in a venv. Every JAX run sets JAX_PLATFORMS=cuda,
so a missing GPU is an error, never a silent CPU run, and the job stops before any arm if JAX sees no GPU. The arms run one at a time on card 0 (J7 on both
cards), J0 first and last; a failed arm is retried once and the job goes on. The host load is logged
every 30 s.
Output in /kaggle/working: results_t4/ (one JSON per arm, J0_b16_start and J0_b16_end), logs/,
hostload.log, env.json.
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
JAX_PINS = ["jax[cuda12]==0.10.2", "optax==0.2.8", "numpy==2.4.6", "ml_dtypes==0.6.0", "scipy==1.17.1"]
CU128 = "https://download.pytorch.org/whl/cu128"
TORCH_PINS = ["torch==2.11.0", "torchvision==0.26.0"]
AURORA_PINS = ["microsoft-aurora==2.0.1", "einops==0.8.2", "timm==1.0.30", "huggingface_hub==2.0.0",
               "safetensors==0.8.0", "numpy==2.4.6", "scipy==1.17.1"]
J, T = "jax", "torch"
ARMS = ([(J, ["--arm", "J0"]), (J, ["--arm", "J1", "--batch", "32"]), (J, ["--arm", "J1", "--batch", "64"])]
        + [(J, ["--arm", a]) for a in ("J2", "J3", "J4", "J5", "J6", "J6t", "fwd", "J7")]
        + [(T, ["--arm", a]) for a in ("T0", "T1", "T2", "T3", "fwd_eager", "fwd_compile", "A0_eager", "A0_compile",
                                       "B0_eager")]
        + [(J, ["--arm", "J0"]), (J, ["--accuracy"])])


def sh(cmd):
    print("+", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
        raise SystemExit(f"failed (rc={r.returncode}): {cmd}")
    return r.stdout


def venv(d):  # the image may lack ensurepip; virtualenv does without it
    return f"({sys.executable} -m venv {d} || ({sys.executable} -m pip install -q virtualenv && {sys.executable} -m virtualenv -q {d}))"


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


code = root_of("experiments/stage12/bench_jax.py", "ugp2_s12_code.tar.gz", "/tmp/code")
data = root_of("data/stage12/ref_e1_cpu.npz", "ugp2_s12_data.tar", "/tmp/data")
for part in ("src", "experiments"):
    shutil.copytree(os.path.join(code, part), os.path.join(W, part), dirs_exist_ok=True)
os.symlink(os.path.join(data, "data"), os.path.join(W, "data"))
commit = open(os.path.join(code, "COMMIT")).read().strip()
os.makedirs(f"{OUT}/logs", exist_ok=True)


def hostload():
    while True:
        with open(f"{OUT}/hostload.log", "a") as f:
            f.write(f"{time.strftime('%FT%T')} load {' '.join(open('/proc/loadavg').read().split()[:3])}\n")
        time.sleep(30)


threading.Thread(target=hostload, daemon=True).start()
py = {J: sys.executable, T: "/tmp/torchenv/bin/python"}
sh(f"{sys.executable} -m pip install -q " + " ".join(f"'{p}'" for p in JAX_PINS))
jax_dev = sh(f"JAX_PLATFORMS=cuda {py[J]} -c 'import jax; d = jax.devices()[0]; print(d.platform, d.device_kind)'").strip()
if not jax_dev.startswith(("gpu", "cuda")):
    raise SystemExit(f"JAX does not see a GPU: {jax_dev}")
sh(venv("/tmp/torchenv") + " && /tmp/torchenv/bin/pip install -q " + " ".join(TORCH_PINS)
   + f" --index-url {CU128} --extra-index-url https://pypi.org/simple")
sh("/tmp/torchenv/bin/pip install -q " + " ".join(AURORA_PINS) + f" --extra-index-url {CU128}")
sh(f"/tmp/torchenv/bin/pip install -q --no-deps --force-reinstall torchvision==0.26.0 --index-url {CU128}")
gpus = [g for g in sh("nvidia-smi -L").splitlines() if g.startswith("GPU")]
torch_ok = sh(f"""{py[T]} -c "import torch, aurora; x = torch.randn(512, 512, device='cuda'); """
              f"""print(torch.__version__, torch.cuda.get_device_name(0), float((x @ x).sum()))" """)
info = dict(commit=commit, gpus=gpus, python=sys.version, jax_device=jax_dev, torch_check=torch_ok.strip(),
            started=time.strftime("%FT%T"),
            jax_versions=sh(f"{py[J]} -m pip freeze").splitlines(), torch_versions=sh(f"{py[T]} -m pip freeze").splitlines())
json.dump(info, open(f"{OUT}/env.json", "w"), indent=1)
print(json.dumps({k: info[k] for k in ("commit", "gpus", "jax_device", "torch_check")}, indent=1), flush=True)

env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false", CUDA_DEVICE_ORDER="PCI_BUS_ID",
           TRITON_CACHE_DIR="/tmp/triton", TORCHINDUCTOR_CACHE_DIR="/tmp/inductor")
RES = f"{W}/experiments/stage12/results"


def run(kind, a, cards="0", tag=""):
    script = "bench_jax.py" if kind == J else "bench_torch.py"
    cmd = [py[kind], f"experiments/stage12/{script}"] + a
    for attempt in (1, 2):
        t0 = time.time()
        e = dict(env, CUDA_VISIBLE_DEVICES=cards, **({"JAX_PLATFORMS": "cuda"} if kind == J else {}))
        with open(f"{OUT}/logs/{'_'.join(a).replace('--', '') or script}{tag}.log", "a") as f:
            rc = subprocess.run(cmd, cwd=W, env=e, stdout=f,
                                stderr=subprocess.STDOUT).returncode
        print(f"{' '.join(a)}{tag}: rc={rc}, {time.time() - t0:.0f} s (try {attempt})", flush=True)
        if rc == 0:
            return True
    return False


if not run(J, ["--arm", "J0", "--smoke", "--steps", "2", "--warmup", "1", "--repeats", "1"], tag="_smoke"):
    raise SystemExit("JAX smoke test failed; see logs/")
print("smoke test passed", flush=True)
for i, (kind, a) in enumerate(ARMS):
    last_j0 = a == ["--arm", "J0"] and i > 0
    run(kind, a, cards="0,1" if a == ["--arm", "J7"] else "0", tag="_end" if last_j0 else "")
    if a == ["--arm", "J0"] and i == 0 and os.path.exists(f"{RES}/J0_b16.json"):
        shutil.copy(f"{RES}/J0_b16.json", f"{RES}/J0_b16_start.json")
    if last_j0 and os.path.exists(f"{RES}/J0_b16.json"):
        shutil.move(f"{RES}/J0_b16.json", f"{RES}/J0_b16_end.json")
        if os.path.exists(f"{RES}/J0_b16_start.json"):
            shutil.copy(f"{RES}/J0_b16_start.json", f"{RES}/J0_b16.json")
    if os.path.isdir(RES):  # saved after every arm, so a session that dies keeps what ran
        shutil.copytree(RES, f"{OUT}/results_t4", dirs_exist_ok=True)
shutil.copytree(RES, f"{OUT}/results_t4", dirs_exist_ok=True)
print(f"done: {len(glob.glob(f'{OUT}/results_t4/*.json'))} result files", flush=True)
