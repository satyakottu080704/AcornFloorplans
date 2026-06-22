# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Converts hand-drawn asbestos-survey sketches into professional Visio (`.vsdx`) floor plans:
walls, doors, windows, numbered rooms, ACM rooms shaded, stair symbols, sample markers, one
page per floor. The production path is fully automated: a surveyor's email → n8n → a Linux
container → SharePoint / AlphaTracker.

## ⚠️ There are TWO independent generators — do not confuse them

This is the single most important thing to understand before editing.

| | **Box pipeline** | **Professional generator (PRODUCTION)** |
|---|---|---|
| Entry | `main.py` → `pipeline.py:process_sketch` | `automation/container/generate_plan.py` (run by `automation/container/process_plan.py`) |
| Geometry source | YOLO **room boxes** in pixel coords (`models/best_floorplans.pt`) + GPT-4o-mini labels | **GPT-4o-mini/OpenAI-first** layout extraction → walls/doors/windows in 0–1000 coords |
| Renderer | COM / Aspose / overlay / XML | `NativeVsdxExporter` (pure XML) or `windows_agent` |
| Used by | local experimentation | **the n8n `plangenration` container** |

The two share almost no code. **Never route the box pipeline's output into the native/Aspose
professional renderer** — box output has no walls and uses pixel coords; forcing it makes the
renderer *invent* walls from approximate boxes, the exact misleading geometry the codebase
guards against (see **Production Status & Decisions** below).

## Commands

```bash
pip install -r requirements.txt          # pywin32 is Windows-only; aspose-diagram covers Linux

# Box pipeline (local) — ONE entry point: main.py. It replaced the old
# run_detector.py + test_local.py (both deleted 2026-06-17).
python main.py --image "sketch.jpg" --output out.vsdx   # YOLO geometry + GPT labels -> Visio
python main.py --image "sketch.jpg" --renderer com      # com needs Visio (clean); aspose = Linux/no-Visio
                                                        # (unlicensed Aspose stamps an "evaluation" watermark)
python main.py --image "sketch.jpg" --overlay           # lock original sketch as background, overlay labels
python main.py --image "sketch.jpg" --no-model          # pure OpenAI vision, skip YOLO (worse geometry — diagnostic)
python main.py --image "sketch.jpg" --model-only        # YOLO-only: detection counts + annotated preview, no API cost
python main.py --batch "folder/" --resume               # batch, resumable
python main.py --clear-cache
# Each run prints a "Geometry source: N from YOLO, N from GPT" line — if GPT
# outvotes YOLO the geometry may not match the sketch (check --model-only).

# Tests (pytest, testpaths=tests; conftest loads .env and puts repo root on sys.path)
pytest -q
pytest tests/test_output_mode.py -q                # one file
pytest tests/test_merge_per_floor.py::test_name -q # one test

# YOLO training
python train_floorplans.py
```

Tests that need `scipy`/`Flask` will error in a bare environment — that's a missing-dep
symptom, not a real failure. Install requirements first.

## Architecture

**Box pipeline (`pipeline.py:process_sketch`)** — preprocess (`utils/room_detection/`) →
YOLO geometry (`yolo_detect_rooms`, class IDs mapped dynamically from `model.names`) →
AI labels via GPT-4o-mini by default (room names, numbers, ACM/no-access flags, sample positions) →
`merge_results` fuses model geometry with AI labels (scipy `linear_sum_assignment` Hungarian
matching by centroid) → render. `merge_results` is the most intricate and regression-prone
function here; it also propagates ACM status from samples to rooms.

**Production generator (`automation/container/generate_plan.py`)** — extracts a
`{walls, doors, windows, rooms, samples}` layout in a **1000×1000 coordinate space**, then
`generate_vsdx_natively` / `NativeVsdxExporter` writes the `.vsdx` as raw OOXML from
`template.vsdx`. It splits Loft/Attic onto its own page (`split_layout_by_floor`); a
visibility guard rejects "microscopic" pages, and `_is_renderable_floor` falls back to one
combined page when a floor is too sparse (otherwise the whole render aborts). Logs
`Layout received -> walls:N …`; **zeros mean an extraction/schema problem**. **Refuses to
render a placeholder** unless `ALLOW_PLACEHOLDER_PLAN=true`.

**The extractor uses one AI provider: OpenAI `gpt-4o-mini`** — `layout_extractor.py:extract_floor_plan_layout` now ignores provider fallback env
vars and calls OpenAI only. This is intentional: GPT-4o-mini is the cheapest reliable model
we measured for this workflow, while Gemini repeatedly returned malformed/truncated JSON on
real sketches.

**Renderers:**
- `NativeVsdxExporter` (`generate_plan.py`) — pure XML, **Windows-free, multi-page. This is
  production.** Not yet at template fidelity (no ACM fill / stairs / title block /
  wall-accurate doors — the open quality work).
- `utils/visio/professional_visio.py` — **COM, needs Windows + Visio**, highest fidelity
  (ACM fill, stairs, title block, legend). The box pipeline's best output.
- `plans/aspose_renderer.py` / `--renderer aspose` — Aspose.Diagram, Linux/no-Visio, but
  **unlicensed = "evaluation" watermark** (looks broken). Not demo-clean without a license.

**Production deployment** — n8n (`automation/n8n/*.json`) polls the Plans inbox → SSHs the
sketch into the `plangenration` container → `process_plan.py "<N>" --image …` gates on status
**Scheduled** (the Wates-client requirement is relaxable via `PLAN_REQUIRE_WATES_CLIENT=false`)
→ draws → uploads the VSDX to a SharePoint **review** folder (AlphaTracker upload + status are
currently disabled). Server `46.62.131.88`, build context `/opt/Plangenration`, shared env
`/opt/acorn/.env`; the container is `FROM acorn_reporting:latest`. Deploy ships the **6
build-context files** (`generate_plan.py`, `process_plan.py`, `layout_extractor.py`,
`vision_client.py`, `template.vsdx`, `Dockerfile`) to `/opt/Plangenration`, then
`docker build` + `docker rm -f` + `docker run`. **See `DOCS.md` (Deployment &
automation runbook) for the exact steps.** `process_plan.py --dry-run` runs the full path with no SharePoint writes.

## Gotchas

- **Plan AI is fixed to OpenAI `gpt-4o-mini`.** `PLAN_LAYOUT_PROVIDERS` remains in old
  env files for compatibility, but the runtime path ignores non-OpenAI providers. Do not
  re-enable Gemini/Groq/Ollama in production without a ground-truth score improvement.
- **`layout_extractor.py` and `vision_client.py` exist in TWO places and must stay in sync:**
  `utils/` (imported by the box pipeline / local runs) and `automation/container/` (the build
  context the Dockerfile COPYs into the container). Edit both, or copy one to the other.
  (The old "don't overwrite the base-image 25 KB extractor" rule is obsolete — the container
  now intentionally uses the repo's `automation/container/layout_extractor.py`.)
- `config.py` couples the YOLO model + class list: `MODEL_PATH`, `MODEL_IMGSZ`, `CLASSES`
  (`acm, door, floor, room, stairs, walls`). Class **order** matters; detection maps IDs from
  `model.names`, so swapping the `.pt` alone is usually safe.
- Model weights/datasets are git-ignored and shipped via **Git LFS** — a clone needs
  `git lfs install && git lfs pull`, else you get pointer files.
- **Cross-platform (Windows dev + Linux container):** no hardcoded per-machine paths (the
  Visio template falls back to the repo's `utils/visio/template.vsdx`); `win32com`/`pywin32`
  and Tesseract are OS-gated; renderer auto-selects (`com` on Windows, native/aspose on Linux).
  `.gitattributes` forces `*.py`/`*.json`/`*.md` to **LF** so the two `gemini_*.py` copies don't
  drift between a Windows checkout and the Linux build.
- The root `README.md` is stale (pure-GPT-4o / ResNet era); trust `config.py`,
  `DOCS.md`, and the code.

## Production Status & Decisions

**Status: deployed but REVIEW-ONLY — not approved for unattended publishing.** A VSDX that
opens in Visio is *not* proof the plan is correct. The blocker is **measured accuracy**, not a
missing feature — so **build no new model/renderer feature until the eval harness gives a
number**.

### The accuracy problem = two separate jobs
- **Geometry** (count rooms / where): the **900-image YOLO model** is trained for this; the LLM
  can under-count (e.g. 4 rooms where the sketch has ~8).
- **Text** (room names, numbers, sample IDs — handwriting): needs an **AI vision model** (GPT-4o).
  Free OCR (Tesseract/Paddle/RapidOCR) **fails on handwriting** (reads names as `Casino`, `BAM`).

A single AI read returns **both** (count + positions + names + samples) in **one call** — no
second read, no double cost.

### Options (decide with the harness, not opinion)
- **Architecture:** A1 = AI does both (production now). A2 = YOLO counts + AI labels (the box
  pipeline `merge_results`) — better *if* the AI under-counts rooms.
- **Text reader:** **GPT-4o-mini ≈ $0.002/sketch — keep (cheap + works).** GPT-4o ≈ $0.035
  (upgrade only if the eval delta earns ~$20/mo). **Gemini — drop** (failed repeatedly:
  quota / timeout / truncated JSON). Free OCR / TrOCR — baseline only.
- **Cost is not a constraint:** ~600 sketches/mo ≈ $1 (mini) to $21 (full GPT-4o).

### Evaluation harness (built — how every change is judged)
- `plans/llm_extract.py` — AI predictor (one read → count + labels), provider via `PLAN_LAYOUT_PROVIDERS`.
- `plans/free_local_extract.py` — free/OCR baseline. `plans/ground_truth_eval.py` — scorer
  (room F1, label/number/floor/sample match, accept ≥ 0.85). `DOCS.md` (Evaluation) — workflow.
- The **900 YOLO labels already provide geometry ground truth**; you only hand-label
  names/floors/samples for ~5–10 sketches.

### What needs doing — IN ORDER (do not skip to renderer polish)
1. **Ground-truth set** (5–10 sketches; correct the GPT-4o-mini drafts).
2. **Baseline** measurement (AI-only).
3. **Architecture** decision (A1 vs A2) on the numbers.
4. **Model** decision (mini vs full) on the numbers.
5. **Then** renderer fidelity (ACM fill, stairs, title block, door arcs) — only once accuracy is acceptable.

### Open issues
- LLM geometry accuracy unverified (under-counts); text accuracy unverified (no ground-truth names yet).
- Gemini unusable for extraction — OpenAI is the only reliable provider.
- Native renderer fidelity gap (door arcs, ACM/no-access fill, stairs, title block, legend).
- No automatic acceptance gates in n8n — human-review-only.

### Acceptance gates (before unattended production)
**A)** valid VSDX package; **B)** geometry sanity (walls in image bounds, doors on walls, labels
inside enclosed rooms, explicit floors, no dummy geometry); **C)** ground-truth accuracy metrics
improving; **D)** surveyor sign-off on ≥ 30 real sketches. Failures route to `Manual_Review`,
never auto-publish.
