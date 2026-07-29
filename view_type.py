"""
View-type classification for catalog images (Step 2 of the upgrade roadmap).

Problem it fixes: the CLIP similarity search previously compared a
customer's "front" photo against front/sleeve/butta/back images all mixed
together in one vector space, diluting the top-5 results. This module
tags every catalog row with a view_type so search can be restricted to
the relevant view only.

This is a heuristic keyword classifier over messy real-world file names
(see design_database.csv — 1,391 distinct raw file_name values with typos
like "frount", inconsistent spacing, numbering like "s1"/"b2", etc). It
will not be 100% perfect on every naming variant; anything unmatched
falls into "other" rather than being silently misclassified, so you can
audit and improve the keyword list over time (see classify_stats()).
"""

import re
from config import VIEW_TYPE_KEYWORDS

# Matches short designer shorthand tokens, e.g. "s1"/"s2" -> sleeve,
# "b1".."b3" -> back, "h1"/"h2"/"hs" -> sleeve (hand), "f"/"f1" -> front.
# These are checked per-token (split on space/dash) so they also catch
# forms like "B1-LC 1689" or "LN Dgn - 0572 F".
_SLEEVE_SHORTHAND = re.compile(r"^(s\d*|h\d*[a]?|hs)$")
_BACK_SHORTHAND = re.compile(r"^b\d*$")
_FRONT_SHORTHAND = re.compile(r"^(f\d*|fr\d*)$")

_TOKEN_SPLIT = re.compile(r"[\s\-_]+")


def classify_view_type(file_name: str) -> str:
    """Classify a single catalog file_name into a view type.

    Returns one of: 'butta', 'sleeve', 'back', 'front', 'full', 'other'.
    """
    if not isinstance(file_name, str) or not file_name.strip():
        return "other"

    name = file_name.lower().replace(".dst", "").strip()

    # Check specific keyword groups first (order defined in config.py),
    # skipping "full" on the first pass since "full sleeve" etc. should
    # be classified as sleeve, not full.
    for view_type in ["butta", "sleeve", "back", "front"]:
        keywords = VIEW_TYPE_KEYWORDS[view_type]
        if any(keyword in name for keyword in keywords):
            return view_type

    # Shorthand codes checked per-token, e.g. "b1-lc 1689" -> ["b1","lc","1689"]
    tokens = [t for t in _TOKEN_SPLIT.split(name) if t]
    for token in tokens:
        if _SLEEVE_SHORTHAND.match(token):
            return "sleeve"
        if _BACK_SHORTHAND.match(token):
            return "back"
        if _FRONT_SHORTHAND.match(token):
            return "front"

    # Fall back to "full" (whole-garment / overview shots)
    if any(keyword in name for keyword in VIEW_TYPE_KEYWORDS["full"]):
        return "full"

    return "other"


def add_view_type_column(df, file_name_col="file_name"):
    """Return a copy of df with a new 'view_type' column added."""
    df = df.copy()
    df["view_type"] = df[file_name_col].apply(classify_view_type)
    return df


def classify_stats(df, file_name_col="file_name", view_type_col="view_type"):
    """Print a quick breakdown of how many rows fell into each view_type,
    and a sample of 'other' rows so you can spot gaps in the keyword list.
    """
    counts = df[view_type_col].value_counts()
    print("View-type breakdown:")
    print(counts.to_string())
    print(f"\nTotal rows: {len(df)}  |  'other' rate: {counts.get('other', 0) / len(df):.1%}")

    other_examples = df.loc[df[view_type_col] == "other", file_name_col].unique()[:20]
    if len(other_examples):
        print("\nSample 'other' file_name values (consider adding keywords for these):")
        for example in other_examples:
            print(f"  - {example}")


if __name__ == "__main__":
    # Standalone migration: add view_type to design_database.csv in place.
    import pandas as pd
    from config import DATABASE_CSV

    print(f"Loading {DATABASE_CSV} ...")
    design_df = pd.read_csv(DATABASE_CSV)

    design_df = add_view_type_column(design_df)
    classify_stats(design_df)

    design_df.to_csv(DATABASE_CSV, index=False)
    print(f"\nSaved view_type column back to {DATABASE_CSV}")
