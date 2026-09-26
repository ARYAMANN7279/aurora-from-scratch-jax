"""Shared setup for Stage 1 experiment scripts."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
DATA = os.path.join(ROOT, "data", "stage1")
RESULTS = os.path.join(HERE, "results")
FIGS = os.path.join(HERE, "figs")


def setup_precision(f32=False):
    import jax

    if not f32:
        jax.config.update("jax_enable_x64", True)


def write_json(name, obj):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    print("wrote", path)


def read_json(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def command_line():
    return " ".join([".venv/bin/python", os.path.relpath(sys.argv[0], ROOT)] + sys.argv[1:])
