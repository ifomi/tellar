"""Scan v15 chunked-transcription telemetry for chunk-SEAM join defects.

The defects we hunt (all live at chunk boundaries, produced by finalize()'s
naive join):

  DUP — a word (or short run) repeated across a seam: whisper transcribes
        the same token at the end of chunk N (it is also in the rolling
        prompt) and again at the start of chunk N+1.
        e.g. "...происходит Происходит транскрибация"
  CAP — chunk N+1 starts with a Capital while chunk N did NOT end in .!?:
        a mid-sentence pause got rendered as a false sentence break, which
        is the "рваность" the user reports. finalize() already fixes the
        inverse (N ends .!?, N+1 lowercase) but not this direction.

Two data sources per record:
  * chunk_texts — logged since the Phase-0 telemetry change OR recovered by
    --replay. Gives EXACT seams → precise DUP/CAP classification.
  * assembled `text` only (older records) — DUP is still precise (adjacent
    duplicate word); CAP is heuristic (can't see seams) → reported as
    low-confidence and restricted to Cyrillic non-vocab words to cut the
    proper-noun false positives (Design Thinking, Fox & Rabbits, ...).

--replay re-runs historical multi-chunk WAVs through the offline harness
(order=suffix, matching v15) to recover chunk_texts, cached in a sidecar
JSONL keyed by id so subsequent runs are instant.

Usage:
  python tools/scan_defects.py                 # fast: heuristics on the log
  python tools/scan_defects.py --replay        # precise: replay WAVs w/o cached seams
  python tools/scan_defects.py --replay --limit 50
  python tools/scan_defects.py --report        # also write the Obsidian report
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SUPPORT = Path(os.path.expanduser("~/Library/Application Support/Tellar"))
LOG = SUPPORT / "transcription_log_chunked.jsonl"
SAMPLES = SUPPORT / "samples"
SEAM_CACHE = SUPPORT / "chunk_texts_cache.jsonl"  # id -> [chunk_texts]
VOCAB = SUPPORT / "vocabulary.txt"
OBSIDIAN = Path(os.path.expanduser("~/Documents/Obsidian Vault/Tellar/Chunk Join Defects.md"))

V15 = "chunked_rolling_v15_suffix_punctuation"

WORD = re.compile(r"[A-Za-zА-Яа-яЁё]+(?:-[A-Za-zА-Яа-яЁё]+)*")
CYR = re.compile(r"[А-Яа-яЁё]")
UPPER = re.compile(r"[A-ZА-ЯЁ]")
TERMINATORS = ".!?"


def load_vocab() -> set:
    v = set()
    if VOCAB.exists():
        for line in VOCAB.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                for w in WORD.findall(line):
                    v.add(w.lower())
    return v


def load_records() -> List[dict]:
    rows = []
    if not LOG.exists():
        return rows
    for line in LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("variant") == V15:
            rows.append(r)
    return rows


def load_seam_cache() -> Dict[int, List[str]]:
    cache: Dict[int, List[str]] = {}
    if SEAM_CACHE.exists():
        for line in SEAM_CACHE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                cache[int(r["id"])] = r["chunk_texts"]
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return cache


def append_seam_cache(rec_id: int, chunk_texts: List[str]):
    SEAM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(SEAM_CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": rec_id, "chunk_texts": chunk_texts},
                           ensure_ascii=False) + "\n")


# ---------- precise classification from per-chunk text ----------

def classify_seams(chunk_texts: List[str], vocab: set) -> List[dict]:
    """Inspect every seam between consecutive non-empty chunks.
    Returns a list of defect dicts {seam, kind, detail}."""
    defects = []
    parts = [(i, t) for i, t in enumerate(chunk_texts) if t and t.strip()]
    for k in range(len(parts) - 1):
        ai, a = parts[k]
        bi, b = parts[k + 1]
        a_s, b_s = a.rstrip(), b.lstrip()
        a_words = WORD.findall(a_s)
        b_words = WORD.findall(b_s)
        if not a_words or not b_words:
            continue
        a_last, b_first = a_words[-1], b_words[0]
        a_terminated = a_s[-1] in TERMINATORS
        # DUP — same word repeated across the seam.
        if len(a_last) >= 3 and a_last.lower() == b_first.lower():
            defects.append({
                "seam": ai, "kind": "DUP",
                "detail": f"...{a_last} | {b_first}...",
            })
        # CAP — capitalized continuation where chunk A did not terminate.
        elif (not a_terminated and UPPER.match(b_first)
              and b_first.lower() not in vocab):
            defects.append({
                "seam": ai, "kind": "CAP",
                "detail": f"...{a_last} | {b_first}...",
            })
    return defects


# ---------- heuristic classification from assembled text only ----------

def classify_flat(text: str, vocab: set) -> List[dict]:
    """No seam info — best-effort over the assembled text.
    DUP is precise (adjacent duplicate). CAP is low-confidence and limited
    to Cyrillic non-vocab words to suppress proper-noun false positives."""
    defects = []
    m = list(WORD.finditer(text))
    for i in range(len(m) - 1):
        a, b = m[i].group(0), m[i + 1].group(0)
        gap = text[m[i].end():m[i + 1].start()]
        if len(a) >= 3 and a.lower() == b.lower():
            defects.append({"kind": "DUP", "conf": "high",
                            "detail": f"...{a} {b}..."})
        elif (UPPER.match(b) and CYR.match(b) and a[-1:].islower()
              and not re.search(r"[.!?:;]", gap)
              and b.lower() not in vocab and len(b) >= 2):
            defects.append({"kind": "CAP", "conf": "low",
                            "detail": f"...{a} {b}..."})
    return defects


def recover_chunk_texts(wav_path: str) -> List[str]:
    """Replay one WAV through the v15 (suffix) pipeline → per-chunk text."""
    from replay_chunked import replay
    chunks = replay(wav_path, order="suffix")
    return [text for (_idx, _reason, _dur, text) in chunks]


def validate():
    """Apply the Phase-2 reconcile_seams to every cached record and report
    how many seam defects it removes, plus before/after text so CAP
    lowercasing can be eyeballed for proper-noun regressions."""
    from tellar.seams import reconcile_seams, vocabulary_word_set
    vocab = vocabulary_word_set()
    cache = load_seam_cache()
    print(f"cached records: {len(cache)} | vocab terms: {len(vocab)}")

    before_dup = before_cap = after_dup = after_cap = 0
    changed = []
    for rid, cts in cache.items():
        b = classify_seams(cts, vocab)
        before_dup += sum(1 for d in b if d["kind"] == "DUP")
        before_cap += sum(1 for d in b if d["kind"] == "CAP")
        fixed = reconcile_seams(cts, vocab)
        a = classify_seams(fixed, vocab)
        after_dup += sum(1 for d in a if d["kind"] == "DUP")
        after_cap += sum(1 for d in a if d["kind"] == "CAP")
        before_txt = " ".join(p for p in cts if p)
        after_txt = " ".join(p for p in fixed if p)
        if before_txt != after_txt:
            changed.append((rid, before_txt, after_txt))

    print("\n=== BEFORE → AFTER (seam defects across all cached records) ===")
    print(f"  DUP occurrences: {before_dup} → {after_dup}")
    print(f"  CAP occurrences: {before_cap} → {after_cap}")
    print(f"  records whose final text changed: {len(changed)}")

    print("\n=== sample diffs (first 25 changed) — eyeball for regressions ===")
    for rid, b, a in changed[:25]:
        # show the words that differ, with a little context
        print(f"\n#{rid}")
        for bw, aw in _word_diffs(b, a):
            print(f"    {bw}  →  {aw}")


def _word_diffs(before: str, after: str, ctx: int = 2):
    """Yield (before_fragment, after_fragment) for each differing word,
    aligned positionally (reconcile only edits/deletes, never reorders)."""
    bw, aw = before.split(), after.split()
    import difflib
    sm = difflib.SequenceMatcher(a=bw, b=aw, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        b_frag = " ".join(bw[max(0, i1 - ctx):i2 + ctx])
        a_frag = " ".join(aw[max(0, j1 - ctx):j2 + ctx])
        out.append((f"…{b_frag}…", f"…{a_frag}…"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", action="store_true",
                    help="recover missing seams by replaying WAVs (slow, cached)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap how many records to replay this run (0 = all)")
    ap.add_argument("--report", action="store_true",
                    help="write the Obsidian markdown report")
    ap.add_argument("--validate", action="store_true",
                    help="apply reconcile_seams to cached seams, report before/after")
    args = ap.parse_args()

    if args.validate:
        validate()
        return

    vocab = load_vocab()
    rows = load_records()
    cache = load_seam_cache()
    print(f"v15 records: {len(rows)} | vocab terms: {len(vocab)} | "
          f"cached seams: {len(cache)}")

    # Optionally replay to fill missing seams.
    if args.replay:
        todo = [r for r in rows
                if r.get("n_chunks", 1) > 1
                and int(r.get("id", -1)) not in cache
                and r.get("wav")]
        if args.limit:
            todo = todo[:args.limit]
        print(f"replaying {len(todo)} multi-chunk WAVs missing seams...")
        for n, r in enumerate(todo, 1):
            rid = int(r["id"])
            wav = SAMPLES / r["wav"]
            if not wav.exists():
                print(f"  [{n}/{len(todo)}] #{rid} SKIP (wav missing)")
                continue
            try:
                cts = recover_chunk_texts(str(wav))
                append_seam_cache(rid, cts)
                cache[rid] = cts
                print(f"  [{n}/{len(todo)}] #{rid} ok ({len(cts)} chunks)")
            except Exception as e:
                print(f"  [{n}/{len(todo)}] #{rid} FAILED: {e}")

    # Classify every record using the best source available.
    findings = []  # {id, n_chunks, source, defects, text}
    for r in rows:
        rid = int(r.get("id", -1)) if r.get("id") is not None else -1
        text = r.get("text") or ""
        cts = r.get("chunk_texts") or cache.get(rid)
        if cts:
            defects = classify_seams(cts, vocab)
            source = "seams"
        elif text:
            defects = classify_flat(text, vocab)
            source = "flat"
        else:
            continue
        if defects:
            findings.append({"id": rid, "n_chunks": r.get("n_chunks"),
                             "source": source, "defects": defects, "text": text})

    # Summary.
    def count(kind, src=None):
        return sum(1 for f in findings
                   if (src is None or f["source"] == src)
                   and any(d["kind"] == kind for d in f["defects"]))

    n_seam = sum(1 for f in findings if f["source"] == "seams")
    n_flat = sum(1 for f in findings if f["source"] == "flat")
    print("\n=== DEFECT SUMMARY ===")
    print(f"records with ≥1 defect: {len(findings)}  "
          f"(precise seams: {n_seam}, heuristic flat: {n_flat})")
    print(f"  DUP total: {count('DUP')}   (precise: {count('DUP','seams')}, "
          f"flat: {count('DUP','flat')})")
    print(f"  CAP total: {count('CAP')}   (precise: {count('CAP','seams')}, "
          f"flat/low-conf: {count('CAP','flat')})")

    # Sorted listing — precise first.
    findings.sort(key=lambda f: (f["source"] != "seams", -(f["n_chunks"] or 0)))
    print("\n=== FLAGGED RECORDS (top 30) ===")
    for f in findings[:30]:
        kinds = ",".join(sorted({d["kind"] for d in f["defects"]}))
        ex = "; ".join(d["detail"] for d in f["defects"][:3])
        print(f"  #{f['id']} [{f['source']:>5}] {kinds:<7} n={f['n_chunks']}  {ex}")

    if args.report:
        write_report(findings, len(rows), n_seam, n_flat, count)
        print(f"\nreport → {OBSIDIAN}")


def write_report(findings, total, n_seam, n_flat, count):
    lines = [
        "# Chunk Join Defects (v15)",
        "",
        f"- v15 records scanned: **{total}**",
        f"- records with ≥1 defect: **{len(findings)}** "
        f"(precise seams: {n_seam}, heuristic flat: {n_flat})",
        f"- DUP: **{count('DUP')}** (precise {count('DUP','seams')} / "
        f"flat {count('DUP','flat')})",
        f"- CAP: **{count('CAP')}** (precise {count('CAP','seams')} / "
        f"flat-low-conf {count('CAP','flat')})",
        "",
        "> `seams` = classified from exact per-chunk text (logged or replayed). "
        "`flat` = heuristic over assembled text; CAP there is low-confidence.",
        "",
        "## Flagged records",
        "",
    ]
    for f in findings:
        kinds = ",".join(sorted({d["kind"] for d in f["defects"]}))
        lines.append(f"### #{f['id']} — {kinds} ({f['source']}, n={f['n_chunks']})")
        for d in f["defects"]:
            lines.append(f"- **{d['kind']}**: `{d['detail']}`")
        if f["text"]:
            lines.append(f"\n> {f['text'][:400]}")
        lines.append("")
    OBSIDIAN.parent.mkdir(parents=True, exist_ok=True)
    OBSIDIAN.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
