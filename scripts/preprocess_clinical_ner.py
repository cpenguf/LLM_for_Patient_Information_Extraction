#!/usr/bin/env python3
"""
Lightweight preprocessing helper to turn clinical NER datasets (2010 i2b2,
2018 n2c2, 2022 n2c2, RadGraph) in CoNLL/BIO format into:
- SFT JSONL for GatorTron/GatorTronLLaMA instruction tuning (`input`, `output`)
- MRC JSON for ClinicalTransformerMRC-style training (query/span format)

Expected input per split: <split>.tsv with tokens and labels (BIO) separated
by tabs, blank lines between sentences.

Outputs (under --out_dir):
- sft/<split>.jsonl
    {"input": "<sentence>", "output": "token label\\n..."}
- mrc/mrc-ner.<split>
    JSON array with entries:
      {"context": "...", "query": "...", "entity_label": "...",
       "start_position": [int...], "end_position": [int...], "impossible": bool}

This is a template; adjust queries/prompts as needed for your task ontology.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional


def read_conll(path: Path) -> List[List[Tuple[str, str]]]:
    sentences, cur = [], []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                if cur:
                    sentences.append(cur)
                    cur = []
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            token, label = parts[0], parts[-1]
            cur.append((token, label))
    if cur:
        sentences.append(cur)
    return sentences


def read_brat(doc_txt: Path, doc_ann: Path) -> List[List[Tuple[str, str]]]:
    """
    Minimal brat reader that converts document-level .ann into a single-sentence BIO sequence.
    Tokens are whitespace-split; character offsets are used to assign BIO labels.
    """
    text = doc_txt.read_text()
    tokens = text.split()
    # build token offsets
    offsets = []
    pos = 0
    for tok in tokens:
        start = text.find(tok, pos)
        end = start + len(tok)
        offsets.append((start, end))
        pos = end

    entities = []
    for line in doc_ann.read_text().splitlines():
        if not line or not line.startswith("T"):
            continue
        try:
            _, span_info, _ = line.split("\t", 2)
            parts = span_info.split()
            label, start, end = parts[0], int(parts[1]), int(parts[2])
            entities.append((start, end, label))
        except ValueError:
            continue

    labels = ["O"] * len(tokens)
    for start, end, label in entities:
        for idx, (tok_s, tok_e) in enumerate(offsets):
            if tok_s >= start and tok_e <= end:
                prefix = "B" if tok_s == start or labels[idx] == "O" else "I"
                labels[idx] = f"{prefix}-{label}"

    return [[(t, l) for t, l in zip(tokens, labels)]]


def spans_from_bio(tokens: List[str], labels: List[str]) -> List[Tuple[int, int, str]]:
    spans = []
    start = None
    cur_label = None
    for i, lab in enumerate(labels):
        if lab.startswith("B-"):
            if start is not None:
                spans.append((start, i - 1, cur_label))
            start, cur_label = i, lab[2:]
        elif lab.startswith("I-"):
            if cur_label != lab[2:]:
                if start is not None:
                    spans.append((start, i - 1, cur_label))
                start, cur_label = i, lab[2:]
        else:
            if start is not None:
                spans.append((start, i - 1, cur_label))
                start, cur_label = None, None
    if start is not None:
        spans.append((start, len(tokens) - 1, cur_label))
    return spans


def token_offsets(tokens: List[str]) -> List[Tuple[int, int]]:
    offsets, pos = [], 0
    for tok in tokens:
        start = pos
        end = start + len(tok)
        offsets.append((start, end))
        pos = end + 1  # assume space separator
    return offsets


def to_sft(sentences: List[List[Tuple[str, str]]]) -> List[Dict]:
    records = []
    for sent in sentences:
        toks, labs = zip(*sent)
        text = " ".join(toks)
        label_lines = "\n".join([f"{t} {l}" for t, l in sent])
        records.append(
            {
                "input": f"Extract clinical entities with BIO tags:\n{text}",
                "output": label_lines,
            }
        )
    return records


def to_mrc(sentences: List[List[Tuple[str, str]]]) -> List[Dict]:
    examples = []
    for sent_id, sent in enumerate(sentences):
        toks, labs = zip(*sent)
        text = " ".join(toks)
        spans = spans_from_bio(list(toks), list(labs))
        offsets = token_offsets(list(toks))
        label_set = sorted({lab for _, _, lab in spans}) if spans else []
        for label in label_set or ["NA"]:
            label_spans = [(s, e) for s, e, l in spans if l == label]
            start_positions = [offsets[s][0] for s, _ in label_spans]
            end_positions = [offsets[e][1] for _, e in label_spans]
            examples.append(
                {
                    "context": text,
                    "query": f"Find all {label} entities.",
                    "entity_label": label,
                    "start_position": start_positions,
                    "end_position": end_positions,
                    "impossible": len(label_spans) == 0,
                    "id": f"{sent_id}_{label}",
                }
            )
    return examples


def write_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_json(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def process_split(split: str, in_dir: Path, out_dir: Path):
    sentences: List[List[Tuple[str, str]]] = []
    conll_path = in_dir / f"{split}.tsv"
    if conll_path.exists():
        sentences = read_conll(conll_path)
    else:
        ann_files = sorted(in_dir.glob(f"{split}*.ann"))
        for ann in ann_files:
            txt = ann.with_suffix(".txt")
            if txt.exists():
                sentences.extend(read_brat(txt, ann))
    if not sentences:
        return
    sft_records = to_sft(sentences)
    mrc_records = to_mrc(sentences)
    write_jsonl(out_dir / "sft" / f"{split}.jsonl", sft_records)
    write_json(out_dir / "mrc" / f"mrc-ner.{split}", mrc_records)


def main():
    parser = argparse.ArgumentParser(description="Preprocess clinical NER datasets into SFT and MRC formats.")
    parser.add_argument("--data_root", required=True, help="Root containing dataset folders (2010_i2b2, 2018_n2c2, 2022_n2c2, radgraph)")
    parser.add_argument("--out_root", required=True, help="Output root for processed data")
    args = parser.parse_args()

    dataset_dirs = ["2010_i2b2", "2018_n2c2", "2022_n2c2", "radgraph"]
    for ds in dataset_dirs:
        in_dir = Path(args.data_root) / ds
        out_dir = Path(args.out_root) / ds
        for split in ["train", "dev", "test"]:
            process_split(split, in_dir, out_dir)


if __name__ == "__main__":
    main()
