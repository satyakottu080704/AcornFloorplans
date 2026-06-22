# Acorn Floor Plan Generation

Converts hand-drawn asbestos-survey sketches into professional Visio (`.vsdx`)
floor plans (walls, doors, numbered rooms, ACM shading, sample markers).

## Run it

```bash
pip install -r requirements.txt
python main.py --image "path/to/sketch.jpg"        # single sketch -> .vsdx
python main.py --batch "path/to/folder" --resume   # batch
python main.py --model-only --image "sketch.jpg"   # YOLO-only diagnostic (no API cost)
```

`main.py` is the single entry point for the local box pipeline (YOLO geometry +
GPT-4o labels → Visio). It prints a `Geometry source: N from YOLO, N from GPT`
line so you can see whether YOLO drove the geometry.

## Documentation

**[CLAUDE.md](CLAUDE.md) is the source of truth** — architecture, the two
generators, production status, cost, and acceptance criteria. Start there.

**[DOCS.md](DOCS.md)** holds everything else in one place — requirements,
deployment/n8n/agent runbooks, evaluation/measurement, renderer convergence,
and historical notes (consolidated from the old scattered `.md` files).

This README is intentionally a short pointer; do not duplicate detail here.
