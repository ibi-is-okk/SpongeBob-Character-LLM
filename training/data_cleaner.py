"""Convert plain-text transcripts you have the right to use into instruction/response JSONL.

Expected raw format (one line per utterance, any number of .txt files):
    SpongeBob: I'm ready! I'm ready!
    Squidward: Be quiet, SpongeBob.
Lines in [brackets] or (parentheses) are stage directions and are dropped.

Usage:
    python training/data_cleaner.py --input data/raw --output data/processed/spongebob.jsonl
"""
import argparse, json, re, random
from pathlib import Path

LINE = re.compile(r"^\s*([A-Za-z][A-Za-z .'\-]{0,30}?)\s*:\s*(.+?)\s*$")
STAGE = re.compile(r"\[[^\]]*\]|\([^)]*\)")

def parse_file(path: Path):
    turns = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = LINE.match(raw)
        if not m:
            continue
        speaker, text = m.group(1).strip(), STAGE.sub("", m.group(2))
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            turns.append((speaker, text))
    return turns

def build_pairs(turns, target, context, min_words, max_words):
    pairs = []
    for i, (speaker, text) in enumerate(turns):
        if speaker.lower() != target.lower() or i == 0:
            continue
        if not (min_words <= len(text.split()) <= max_words):
            continue
        prev = turns[max(0, i - context):i]
        if all(s.lower() == target.lower() for s, _ in prev):
            continue  # no outside speaker to respond to
        instruction = "\n".join(f"{s}: {t}" for s, t in prev)
        pairs.append({"instruction": instruction, "response": text})
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw")
    ap.add_argument("--output", default="data/processed/spongebob.jsonl")
    ap.add_argument("--character", default="SpongeBob")
    ap.add_argument("--context", type=int, default=2, help="preceding turns used as the prompt")
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--max-words", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    files = sorted(Path(a.input).glob("**/*.txt"))
    if not files:
        raise SystemExit(f"No .txt transcripts found in {a.input}. Add your own files first.")
    pairs, seen = [], set()
    for f in files:
        for p in build_pairs(parse_file(f), a.character, a.context, a.min_words, a.max_words):
            key = (p["instruction"], p["response"])
            if key not in seen:
                seen.add(key); pairs.append(p)
    random.Random(a.seed).shuffle(pairs)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pairs)} pairs from {len(files)} files -> {out}")

if __name__ == "__main__":
    main()
