"""
Search logging (Step 4 of the upgrade roadmap).

Every search gets appended as one row to a CSV. This is what feeds:
  - Step 1 (eval set): compare logged predictions against known-correct answers
  - Step 5 (memory review): spot low-confidence / conflicting confirmations
  - Step 9 (CLIP fine-tuning): (photo, correct_design) pairs for training
  - Step 10 (dashboard): accuracy trend over time

Kept deliberately dumb (flat CSV, append-only) so it never becomes a
reason the main app breaks - if logging fails for any reason, callers
should treat it as best-effort and continue.
"""

import csv
import os
from datetime import datetime

from config import BASE_DIR

SEARCH_LOG_CSV = str(BASE_DIR / "search_log.csv")

LOG_FIELDS = [
    "timestamp",
    "ocr_code_found",
    "ocr_code_value",
    "catalogue_match_found",
    "memory_match_found",
    "memory_match_score",
    "top1_design_name",
    "top1_designer",
    "top1_confidence_score",
    "top1_match_count",
    "view_type_filter",
    "designer_filter",
    "final_selected_design",  # filled in later if/when staff confirms one
]


def _ensure_log_exists():
    if not os.path.exists(SEARCH_LOG_CSV):
        with open(SEARCH_LOG_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            writer.writeheader()


def log_search(
    ocr_code_found=False,
    ocr_code_value="",
    catalogue_match_found=False,
    memory_match_found=False,
    memory_match_score=None,
    top1_design_name="",
    top1_designer="",
    top1_confidence_score=None,
    top1_match_count=None,
    view_type_filter=None,
    designer_filter=None,
    final_selected_design="",
):
    """Append one row describing a search event. Best-effort: never raises
    to the caller so a logging bug can't take down the actual search flow.
    """
    try:
        _ensure_log_exists()
        with open(SEARCH_LOG_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            writer.writerow({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "ocr_code_found": ocr_code_found,
                "ocr_code_value": ocr_code_value,
                "catalogue_match_found": catalogue_match_found,
                "memory_match_found": memory_match_found,
                "memory_match_score": memory_match_score if memory_match_score is not None else "",
                "top1_design_name": top1_design_name,
                "top1_designer": top1_designer,
                "top1_confidence_score": top1_confidence_score if top1_confidence_score is not None else "",
                "top1_match_count": top1_match_count if top1_match_count is not None else "",
                "view_type_filter": ",".join(view_type_filter) if view_type_filter else "",
                "designer_filter": ",".join(designer_filter) if designer_filter else "",
                "final_selected_design": final_selected_design,
            })
    except Exception as exc:  # noqa: BLE001 - logging must never break the app
        print(f"[search_log] Warning: failed to log search event: {exc}")


def summarize_log(log_path=SEARCH_LOG_CSV):
    """Quick CLI summary: OCR/catalogue/memory hit rates, avg confidence."""
    import pandas as pd

    if not os.path.exists(log_path):
        print(f"No log file found at {log_path} yet - run some searches first.")
        return

    df = pd.read_csv(log_path)
    if df.empty:
        print("Log file exists but has no rows yet.")
        return

    total = len(df)
    print(f"Total logged searches: {total}")
    print(f"OCR code found:        {df['ocr_code_found'].sum()} ({df['ocr_code_found'].mean():.1%})")
    print(f"Catalogue match found: {df['catalogue_match_found'].sum()} ({df['catalogue_match_found'].mean():.1%})")
    print(f"Memory match found:    {df['memory_match_found'].sum()} ({df['memory_match_found'].mean():.1%})")

    conf = pd.to_numeric(df["top1_confidence_score"], errors="coerce").dropna()
    if len(conf):
        print(f"Avg top-1 confidence:  {conf.mean():.1%}  (min {conf.min():.1%}, max {conf.max():.1%})")

    confirmed = df[df["final_selected_design"].astype(str).str.strip() != ""]
    if len(confirmed):
        agree = (confirmed["top1_design_name"] == confirmed["final_selected_design"]).mean()
        print(f"Top-1 == staff-confirmed design: {agree:.1%}  (n={len(confirmed)})")


if __name__ == "__main__":
    summarize_log()
