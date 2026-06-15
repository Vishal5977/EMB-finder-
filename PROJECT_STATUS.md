# KRISHNA EMBROIDERY FINDER — PROJECT STATUS SUMMARY (UPDATED)

## GOAL
AI tool: customer brings an image -> find matching DST embroidery file from ~20,000 designs.

## ENVIRONMENT (FIXED)
- Python 3.11 via `py -3.11` (NOT Anaconda/Ubuntu)
- Project folder: C:\Users\Asus\embroidery-finder\
- Tesseract: C:\Program Files\Tesseract-OCR\tesseract.exe
- Main design library: C:\Users\Asus\all pendrive design\
- RULE: All code via .py files in Notepad, run with: py -3.11 "path\to\script.py"

## ARCHITECTURE STATUS
- Phase 1.1 (OCR): Working ~90% on Krishna codes. Extracts numbers from ANY
  folder name format (e.g. "SVC_5385_BIG_DST" -> matches "5385").
  Known failures: rotated/diagonal text (OCR misreads digits).
- Phase 1.2 (online catalogue): NOT STARTED.
- Phase 1.3 (CLIP visual search): Working as "Top 5 suggestions" tool
  (DECISION: not expecting #1-perfect match, staff picks from 5 candidates).

## KEY FILES (C:\Users\Asus\embroidery-finder\)
- scan_dst.py: scans designer folder -> DST to PNG -> appends to design_database.csv
  Edit DESIGNS_FOLDER + DESIGNER_NAME before each run. Auto-skips already-indexed.
- fingerprint.py: CLIP embeddings (ViT-L-14, 768-dim) on EDGE-DETECTED images,
  saves to Qdrant (qdrant_db/). Re-run after every scan_dst.py.
  If changing model/vector-size: must delete qdrant_db folder first:
  rmdir /s /q "C:\Users\Asus\embroidery-finder\qdrant_db"
- ocr_module.py: Phase 1.1 logic. Number-extraction from design_name lookup.
- app.py: Flow = Upload -> Rotate slider -> Crop tool (streamlit_cropper,
  free-form) -> OCR check -> if no code, CLIP visual search on edge-detected
  cropped image -> top 5 results.
  Run: py -3.11 -m streamlit run "C:\Users\Asus\embroidery-finder\app.py"
- design_database.csv: dst_path, image_path, design_name, file_name, designer

## DATABASE STATUS
- Krishna offer design: 50 files (designer='Krishna')
- Bindu Design: 496 files (designer='Bindu') - no original JPGs
- Varsha Creations: 427 files (designer='Varsha Creations')
- TOTAL: 973 designs indexed + fingerprinted (edge-detection, ViT-L-14)

## KEY LEARNINGS
- ViT-L-14 (768-dim) better than ViT-B-32 (512-dim)
- Edge-detection helps SOME images significantly (+10-14%), not universal
- CROPPING to just embroidery improves scores (e.g. 76.7% -> 85.9% with edges)
- Scores still cluster ~1% spread among top 5 - CLIP's general training limits
  fine discrimination for this domain
- DECISION: Accept as "Top 5 suggestions" tool, staff picks correct one -
  still huge time savings vs manual search of 1000s of files

## NEXT STEPS (priority order)
1. Start DAILY USE for real customers - gather feedback
2. Continue scaling: add designer folders incrementally (clean folder ->
   edit scan_dst.py DESIGNS_FOLDER+DESIGNER_NAME -> run scan_dst.py -> fingerprint.py)
3. Phase 1.2 - not started, future work
4. Future: re-rank by code-frequency in top-N raw results (if same code's
   front/back/sleeve/butta all appear in top 20, boost that code) - discussed
   but not built
5. Fine-tuning CLIP (expensive/complex) - only if daily use proves insufficient

## KNOWN EDGE CASES
- "2018": OCR fails (diagonal text), relies on visual fallback
- "49" (Varsha): never found in database - check folder naming, not checked yet
- "5385"/"SVC 5385": OCR misreads digits (rotation+watermark) - accepted
- Same-code-different-designer: SOLVED via 'designer' column