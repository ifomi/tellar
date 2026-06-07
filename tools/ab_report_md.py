"""Generate the Obsidian Polish-comparison doc from ab_results.jsonl.

Produces: a per-model table (Было | Стало) for each model + one wide summary
("Сводная") table with a column per model. Re-run after adding models with
llm_ab.py — the doc always reflects the current jsonl. Reads local files only
(no network, no GPU):

    cd ~/tellar && .venv/bin/python tools/ab_report_md.py
    cd ~/tellar && .venv/bin/python tools/ab_report_md.py --round v2
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent


def round_paths(round_name, prompt_tag=""):
    """(results_path, samples_dir, output_md) for a named round + prompt.
    Default round '' uses legacy paths. prompt_tag (e.g. 'pv2') appends
    suffixes so a different system prompt has its own jsonl + report doc."""
    if not round_name:
        results_base = HERE / "ab_results"
        samples_dir = HERE / "ab_samples"
        out_name = "Polish — сравнение моделей"
    else:
        results_base = HERE / f"ab_results_{round_name}"
        samples_dir = HERE / f"ab_samples_{round_name}"
        out_name = f"Polish — сравнение моделей {round_name}"
    suffix = f"_{prompt_tag}" if prompt_tag else ""
    out_suffix = f" (prompt {prompt_tag})" if prompt_tag else ""
    return (
        results_base.with_name(results_base.name + suffix + ".jsonl"),
        samples_dir,
        Path.home() / "Documents" / "Obsidian Vault" / "Tellar"
        / f"{out_name}{out_suffix}.md",
    )

# Preferred section/column order; models not listed are appended after.
MODEL_ORDER = [
    "mlx-community/gemma-3-4b-it-qat-4bit",
    "mlx-community/gemma-3-4b-it-4bit",
    "mlx-community/gemma-2-9b-it-4bit",
    "mlx-community/aya-expanse-8b-4bit",
    "mlx-community/Llama-3.1-8B-Instruct-4bit",
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit",
    "mlx-community/Hermes-3-Llama-3.1-8B-4bit",
    "mlx-community/Ministral-8B-Instruct-2410-4bit",
    "mlx-community/Qwen2.5-3B-Instruct-4bit",
]


def short(repo):
    return repo.split("/")[-1]


def cell(text):
    """Make text safe for a single markdown table cell."""
    # Strip chat-template special tokens that leaked into stored outputs
    # (e.g. Gemma-2's trailing "<end_of_turn>").
    for tok in ("<end_of_turn>", "<eos>", "<bos>", "<|im_end|>", "<|eot_id|>",
                "</s>", "<pad>"):
        text = text.replace(tok, "")
    return text.replace("|", "\\|").replace("\n", " ").strip()


def num(sid):
    return sid.split("_")[0]


def lang(sid):
    """Sample language. EN if filename contains an `_en` token, else RU.
    Special-case 05_english (older naming, predates the prefix convention)."""
    parts = sid.split("_")
    if "en" in parts or sid == "05_english":
        return "en"
    return "ru"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="",
                    help="round name; reads ab_results_<name>.jsonl + "
                    "ab_samples_<name>/, writes 'Polish — сравнение моделей "
                    "<name>.md' (default: legacy paths)")
    ap.add_argument("--prompt", default="v1", choices=["v1", "v2"],
                    help="POLISH preset version: v1 = legacy jsonl; "
                    "v2 reads ab_results_<round>_pv2.jsonl and writes a "
                    "separate '(prompt pv2)' report doc")
    args = ap.parse_args()

    prompt_tag = "" if args.prompt == "v1" else "pv2"
    RESULTS, SAMPLES_DIR, OUT = round_paths(args.round, prompt_tag)

    recs = [json.loads(l) for l in RESULTS.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    by = {(r["model"], r["sample"]): r for r in recs}

    models = sorted(
        {r["model"] for r in recs},
        key=lambda m: (MODEL_ORDER.index(m) if m in MODEL_ORDER else 999, m),
    )
    samples = [f.stem for f in sorted(SAMPLES_DIR.glob("*.txt"))]

    # Input text per sample (take from whichever model has it).
    inputs = {}
    for sid in samples:
        for m in models:
            if (m, sid) in by:
                inputs[sid] = by[(m, sid)]["input"]
                break

    out = []
    out.append("# Polish — сравнение моделей (A/B)")
    out.append("")
    out.append("Studio-пресет **Polish** на блоке текстов с типичными артефактами "
               "распознавания (смещённая/пропущенная пунктуация, кривые падежи, "
               "филлеры, самоперебивы). Цель — модель, которая работает со **смыслом**.")
    out.append("")
    out.append("- temperature 0.2, max_tokens 1024 · запуск `tools/llm_ab.py` · "
               "документ генерится `tools/ab_report_md.py`")
    out.append(f"- Моделей в прогоне: **{len(models)}** · текстов: **{len(samples)}**")
    out.append("")

    # --- model summary (avg time) ---
    out.append("## Сводка")
    out.append("")
    out.append("| Модель | Среднее время / текст |")
    out.append("|---|---|")
    for m in models:
        times = [by[(m, s)]["secs"] for s in samples if (m, s) in by]
        avg = sum(times) / len(times) if times else 0.0
        out.append(f"| `{short(m)}` | {avg:.2f} с |")
    out.append("")

    # --- wide summary tables, split by language ---
    ru_samples = [s for s in samples if lang(s) == "ru"]
    en_samples = [s for s in samples if lang(s) == "en"]

    def write_wide(title, sample_ids):
        if not sample_ids:
            return
        out.append(f"## {title}")
        out.append("")
        out.append("| № | Было | " + " | ".join(short(m) for m in models) + " |")
        out.append("|---|---|" + "|".join(["---"] * len(models)) + "|")
        for sid in sample_ids:
            row = [num(sid), cell(inputs.get(sid, ""))]
            for m in models:
                r = by.get((m, sid))
                row.append(cell(r["output"]) if r else "—")
            out.append("| " + " | ".join(row) + " |")
        out.append("")

    write_wide("Сводная таблица — RU", ru_samples)
    write_wide("Сводная таблица — EN", en_samples)

    # --- per-model tables ---
    for m in models:
        out.append(f"## {short(m)}")
        out.append("")
        out.append("| № | Было | Стало |")
        out.append("|---|---|---|")
        for sid in samples:
            r = by.get((m, sid))
            out.append(f"| {num(sid)} | {cell(inputs.get(sid, ''))} | "
                       f"{cell(r['output']) if r else '—'} |")
        out.append("")

    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"  {len(models)} models, {len(samples)} samples")


if __name__ == "__main__":
    main()
