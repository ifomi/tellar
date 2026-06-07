"""Studio-LLM A/B harness — compare candidate models on the Polish preset.

Runs the Polish preset (tellar.studio_llm.POLISH) over every text in
tools/ab_samples/*.txt through each model in MODELS, records results to
tools/ab_results.jsonl, and prints a side-by-side report grouped by sample so
you can eyeball which model actually understands and rewrites the text (vs.
shuffling words cosmetically).

Run from the project venv (HF is reachable from your terminal, not the Claude
sandbox). Models download on first use into the HF cache, then stay cached.

    cd ~/tellar
    .venv/bin/python tools/llm_ab.py                 # run all models in MODELS
    .venv/bin/python tools/llm_ab.py -m gemma-3-4b-it-qat-4bit   # one model
    .venv/bin/python tools/llm_ab.py --report        # just reprint last results

Results accumulate in ab_results.jsonl keyed by (model, sample): re-running a
model overwrites its rows, running a new model adds its rows, so the report
always shows the latest of every model you've tried — run them one at a time
and compare whenever you like.
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

# Make `tellar` importable when run as `python tools/llm_ab.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tellar import studio_llm  # noqa: E402

# Full comparison slate. Use --only-missing to fill in gaps when new samples
# or new models are added without re-running existing pairs (re-runs at
# temperature>0 introduce noise that shifts already-evaluated cells).
MODELS = [
    "mlx-community/gemma-3-4b-it-qat-4bit",
    "mlx-community/gemma-3-4b-it-4bit",
    "mlx-community/gemma-2-9b-it-4bit",
    "mlx-community/aya-expanse-8b-4bit",
    "mlx-community/Llama-3.1-8B-Instruct-4bit",
    "mlx-community/Ministral-8B-Instruct-2410-4bit",
    "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit",
    "mlx-community/Hermes-3-Llama-3.1-8B-4bit",
]

HERE = Path(__file__).parent
SAMPLES_DIR = HERE / "ab_samples"
RESULTS_PATH = HERE / "ab_results.jsonl"


def round_paths(round_name, prompt_tag=""):
    """Map a round name (e.g. 'v2') to (samples_dir, results_path).
    Default round is "" (legacy paths: ab_samples/, ab_results.jsonl).
    prompt_tag (e.g. 'pv2') appends a suffix to the results jsonl so a
    different system prompt writes to its own file without overwriting."""
    if not round_name:
        samples_dir = HERE / "ab_samples"
        base = HERE / "ab_results"
    else:
        samples_dir = HERE / f"ab_samples_{round_name}"
        base = HERE / f"ab_results_{round_name}"
    suffix = f"_{prompt_tag}" if prompt_tag else ""
    return samples_dir, base.with_name(base.name + suffix + ".jsonl")


def load_samples():
    """All sample texts, ordered by filename (the NN_ prefix sets the order)."""
    files = sorted(SAMPLES_DIR.glob("*.txt"))
    return [(f.stem, f.read_text(encoding="utf-8").strip()) for f in files]


def load_results():
    """Existing results as {(model, sample): record}."""
    out = {}
    if RESULTS_PATH.exists():
        for line in RESULTS_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[(rec["model"], rec["sample"])] = rec
    return out


def save_results(results):
    """Write the {(model, sample): record} map back, sorted for stable diffs."""
    rows = sorted(results.values(), key=lambda r: (r["model"], r["sample"]))
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_model(repo, samples, preset):
    """Load `repo`, run `preset` over every sample, return a list of records.
    Frees the model afterwards so models can be run back-to-back on 16 GB."""
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    print(f"\n=== loading {repo} ===", flush=True)
    t_load = time.time()
    model, tok = load(repo)
    print(f"    loaded in {time.time() - t_load:.1f}s", flush=True)

    sampler = make_sampler(temp=preset.temperature)
    records = []
    for sid, text in samples:
        prompt = studio_llm.build_chat_prompt(tok, preset.system, text)
        t0 = time.time()
        raw = generate(model, tok, prompt, max_tokens=preset.max_tokens,
                       sampler=sampler, verbose=False)
        secs = time.time() - t0
        result = studio_llm._strip_preamble(raw)
        records.append({
            "model": repo,
            "sample": sid,
            "secs": round(secs, 2),
            "in_chars": len(text),
            "out_chars": len(result),
            "input": text,
            "output": result,
        })
        print(f"    {sid:14} {secs:5.1f}s  {len(text):4}→{len(result):4} chars", flush=True)

    del model, tok
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass
    return records


def print_report(results, samples):
    """Side-by-side, grouped by sample: input once, then each model's output."""
    by_sample = {}
    for rec in results.values():
        by_sample.setdefault(rec["sample"], []).append(rec)

    for sid, _ in samples:
        recs = sorted(by_sample.get(sid, []), key=lambda r: r["model"])
        if not recs:
            continue
        print("\n" + "#" * 78)
        print(f"# SAMPLE: {sid}")
        print("#" * 78)
        print("INPUT:")
        print(f"  {recs[0]['input']}")
        for rec in recs:
            short = rec["model"].split("/")[-1]
            print(f"\n--- {short}  ({rec['secs']}s) ---")
            print(f"  {rec['output']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--models", help="comma-separated repo ids or short "
                    "names (mlx-community/ prepended); overrides MODELS")
    ap.add_argument("--round", default="",
                    help="round name; selects ab_samples_<name>/ + "
                    "ab_results_<name>.jsonl (default: legacy paths)")
    ap.add_argument("--prompt", default="v1", choices=["v1", "v2"],
                    help="POLISH preset version: v1 = studio_llm.POLISH; "
                    "v2 = studio_llm.POLISH_V2 (writes to "
                    "ab_results_<round>_pv2.jsonl)")
    ap.add_argument("--only-missing", action="store_true",
                    help="skip (model, sample) pairs already in results jsonl")
    ap.add_argument("--report", action="store_true",
                    help="only reprint the report from results jsonl")
    args = ap.parse_args()

    preset = studio_llm.POLISH if args.prompt == "v1" else studio_llm.POLISH_V2
    prompt_tag = "" if args.prompt == "v1" else "pv2"

    global SAMPLES_DIR, RESULTS_PATH
    SAMPLES_DIR, RESULTS_PATH = round_paths(args.round, prompt_tag)

    samples = load_samples()
    if not samples:
        print(f"No samples in {SAMPLES_DIR}/ — add some *.txt files.")
        return
    results = load_results()

    if not args.report:
        if args.models:
            models = [m if "/" in m else f"mlx-community/{m}"
                      for m in args.models.split(",") if m.strip()]
        else:
            models = MODELS
        for repo in models:
            todo = samples
            if args.only_missing:
                todo = [(sid, t) for sid, t in samples
                        if (repo, sid) not in results]
                if not todo:
                    print(f"\n=== {repo}: all samples done, skipping ===",
                          flush=True)
                    continue
                skipped = len(samples) - len(todo)
                if skipped:
                    print(f"\n=== {repo}: {skipped} cached, running "
                          f"{len(todo)} ===", flush=True)
            for rec in run_model(repo, todo, preset):
                results[(rec["model"], rec["sample"])] = rec
            save_results(results)

    print_report(results, samples)
    print(f"\nResults: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
