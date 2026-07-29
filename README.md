# EMB-Finder

AI tool: a customer brings an embroidery design photo → the tool finds the matching DST embroidery file from thousands of indexed designs.

## Pipeline

```
Customer photo
   ↓
Rotate slider (manual correction)
   ↓
Crop tool (isolate just the embroidery)
   ↓
1. OCR (pytesseract) → exact lookup if a design code is readable
   ↓ (if no code)
2. Catalogue photo match (SIFT keypoint matching) → exact photo match
   ↓ (if no match)
3. Match memory (past staff-confirmed matches) → reuse prior corrections
   ↓ (if no match)
4. CLIP (ViT-L-14) + Qdrant vector search → top-5 visual matches, re-ranked
   by a confidence score that rewards a design matching multiple times
   across views/crops (see group_visual_results in app.py)
```

## Setup

1. Install dependencies: `pip install -r requirements.txt` (see below)
2. Install Tesseract OCR (Windows: default path is picked up automatically;
   Linux/Mac: `apt install tesseract-ocr` / `brew install tesseract`, make
   sure it's on PATH)
3. All data paths default to living next to the code (see `config.py`).
   Override with environment variables if you keep data elsewhere:
   `EMB_QDRANT_PATH`, `EMB_DATABASE_CSV`, `EMB_MEMORY_CSV`, `EMB_DST_IMAGES_DIR`,
   `EMB_TESSERACT_CMD`.
4. Run: `streamlit run app.py`

## Key files

- `config.py` — central config (paths, thresholds, re-ranking tuning)
- `view_type.py` — classifies each catalog image as front/sleeve/butta/back/full/other
  from its filename, so search can be restricted to the relevant view
- `search_log.py` — logs every search to `search_log.csv` for accuracy tracking
- `app.py` — the Streamlit app (main entry point)
- `scan_dst.py` — ingests a new designer's DST folder into `design_database.csv`
- `fingerprint.py` — builds/updates CLIP embeddings in `qdrant_db/`.
  Run `python fingerprint.py --rebuild` after any change to view-type
  classification logic so the vector DB payloads stay in sync.
- `design_database.csv` — master index: `dst_path, image_path, design_name, file_name, designer, view_type`
- `catalogue_matcher.py` — SIFT-based exact photo matching (tier 2 of the pipeline)

## Re-ranking

`group_visual_results()` in `app.py` boosts a design's confidence score by
`REPEAT_MATCH_BONUS` (default 1%) for each extra corroborating hit in the
raw top-N results, capped at `REPEAT_MATCH_BONUS_CAP` (default 5) extra
hits. Tune both in `config.py`.

## Search logging

Every search appends a row to `search_log.csv` (OCR/catalogue/memory hit
flags, top-1 design + confidence, filters used). Run
`python search_log.py` for a quick summary (hit rates, average confidence).
This log is the foundation for building an evaluation set and eventually
fine-tuning CLIP on your own confirmed matches.
