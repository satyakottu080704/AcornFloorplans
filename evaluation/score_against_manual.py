#!/usr/bin/env python3
"""
Score the pipeline against the team's manual plans (structural accuracy).

Pairs an INPUT sketch (pulled from AlphaTracker into output/_at_sketches/) with
the manual-plan answer key (ground_truth/manual_plans_truth.json), runs the box
pipeline on the sketch, and compares the EXTRACTED rooms/samples to the answer
key. Uses STRUCTURAL metrics only (count, label, number, sample, floor) — the
manual plan and the sketch are different coordinate spaces, so bbox-IoU is N/A.

Usage:
    python evaluation/score_against_manual.py                 # all available pairs
    python evaluation/score_against_manual.py N-104621 N-105325
"""
import os, sys, json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
except Exception:
    pass

TRUTH = json.load(open(ROOT / "ground_truth" / "manual_plans_truth.json", encoding="utf-8"))["projects"]
SKETCH_DIR = ROOT / "output" / "_at_sketches"


def _norm(s):
    return "".join(c for c in str(s or "").lower() if c.isalnum())


def _multiset_rate(pred, gt):
    """Fraction of gt items matched in pred (order-free, multiset)."""
    pc, gc = Counter(_norm(x) for x in pred if _norm(x)), Counter(_norm(x) for x in gt if _norm(x))
    if not gc:
        return 1.0 if not pc else 0.0
    matched = sum(min(pc[k], gc[k]) for k in gc)
    return matched / sum(gc.values())


def score_one(pn):
    sketch = SKETCH_DIR / f"{pn}_sketch.jpg"
    if not sketch.exists():
        return {"project": pn, "error": "no sketch"}
    truth = TRUTH.get(pn)
    if not truth:
        return {"project": pn, "error": "no answer key"}

    from pipeline import process_sketch
    out = ROOT / "output" / "visio" / f"{pn}_eval.vsdx"
    _, plan = process_sketch(str(sketch), output_path=str(out))

    pred_labels = [r.label for r in plan.rooms]
    pred_numbers = [r.number for r in plan.rooms]
    pred_samples = [s.id for s in plan.samples]
    gt_labels = [r["label"] for r in truth["rooms"]]
    gt_numbers = [r["room_number"] for r in truth["rooms"]]
    gt_samples = truth.get("samples", [])

    return {
        "project": pn,
        "rooms_pred": len(plan.rooms),
        "rooms_gt": len(truth["rooms"]),
        "label_rate": round(_multiset_rate(pred_labels, gt_labels), 2),
        "number_rate": round(_multiset_rate(pred_numbers, gt_numbers), 2),
        "samples_pred": len(pred_samples),
        "samples_gt": len(gt_samples),
        "sample_rate": round(_multiset_rate(pred_samples, gt_samples), 2),
    }


def main():
    pns = sys.argv[1:] or [p for p in TRUTH if (SKETCH_DIR / f"{p}_sketch.jpg").exists()]
    rows = []
    for pn in pns:
        print(f"\n--- scoring {pn} ---")
        try:
            rows.append(score_one(pn))
        except Exception as e:
            rows.append({"project": pn, "error": f"{type(e).__name__}: {e}"})

    print("\n" + "=" * 78)
    print(f"{'Project':<11}{'rooms p/gt':<12}{'label':<8}{'number':<8}{'samp p/gt':<11}{'sample':<8}")
    print("-" * 78)
    for r in rows:
        if r.get("error"):
            print(f"{r['project']:<11}ERROR: {r['error']}")
            continue
        print(f"{r['project']:<11}{str(r['rooms_pred'])+'/'+str(r['rooms_gt']):<12}"
              f"{r['label_rate']:<8}{r['number_rate']:<8}"
              f"{str(r['samples_pred'])+'/'+str(r['samples_gt']):<11}{r['sample_rate']:<8}")
    ok = [r for r in rows if not r.get("error")]
    if ok:
        print("-" * 78)
        print(f"{'MEAN':<11}{'':<12}{sum(r['label_rate'] for r in ok)/len(ok):<8.2f}"
              f"{sum(r['number_rate'] for r in ok)/len(ok):<8.2f}{'':<11}"
              f"{sum(r['sample_rate'] for r in ok)/len(ok):<8.2f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
