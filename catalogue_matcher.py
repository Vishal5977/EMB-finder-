"""Photo-to-photo catalogue matching for embroidery designs."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

CATALOGUE_CSV = Path(r"C:\Users\Asus\embroidery-finder\catalogue_database.csv")
MAX_IMAGE_SIDE = 1200
MIN_GOOD_MATCHES = 10
MIN_INLIERS = 8
MIN_INLIER_RATIO = 0.25


def _prepare_gray(image):
    if isinstance(image, (str, Path)):
        image = Image.open(image)

    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    scale = min(1.0, MAX_IMAGE_SIDE / max(width, height))
    if scale < 1.0:
        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    return gray


def _compare_images(query_gray, reference_gray):
    sift = cv2.SIFT_create(
        nfeatures=3000,
        contrastThreshold=0.02,
        edgeThreshold=15,
    )
    query_points, query_descriptors = sift.detectAndCompute(query_gray, None)
    reference_points, reference_descriptors = sift.detectAndCompute(reference_gray, None)

    if query_descriptors is None or reference_descriptors is None:
        return 0, 0, 0.0
    if len(query_descriptors) < 2 or len(reference_descriptors) < 2:
        return 0, 0, 0.0

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(query_descriptors, reference_descriptors, k=2)
    good_matches = [
        first for first, second in pairs
        if first.distance < 0.72 * second.distance
    ]

    if len(good_matches) < 4:
        return len(good_matches), 0, 0.0

    query_locations = np.float32([
        query_points[match.queryIdx].pt for match in good_matches
    ]).reshape(-1, 1, 2)
    reference_locations = np.float32([
        reference_points[match.trainIdx].pt for match in good_matches
    ]).reshape(-1, 1, 2)

    _, inlier_mask = cv2.findHomography(
        query_locations,
        reference_locations,
        cv2.RANSAC,
        5.0,
    )
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    inlier_ratio = inliers / len(good_matches) if good_matches else 0.0
    return len(good_matches), inliers, inlier_ratio


def find_catalogue_match(image, selected_designers=None):
    """Return the best verified catalogue match, or None."""
    if not CATALOGUE_CSV.exists():
        return None

    catalogue = pd.read_csv(CATALOGUE_CSV)
    if selected_designers:
        selected_designers = {str(designer).strip() for designer in selected_designers}
        catalogue = catalogue[
            catalogue["designer"].astype(str).str.strip().isin(selected_designers)
        ]
        if catalogue.empty:
            return None

    query_gray = _prepare_gray(image)
    candidates = []

    for _, row in catalogue.iterrows():
        reference_path = Path(str(row["image_path"]))
        if not reference_path.exists():
            continue

        reference_gray = _prepare_gray(reference_path)
        good_matches, inliers, inlier_ratio = _compare_images(
            query_gray,
            reference_gray,
        )
        candidates.append({
            "designer": str(row["designer"]),
            "design_name": str(row["design_name"]),
            "image_path": str(reference_path),
            "good_matches": good_matches,
            "inliers": inliers,
            "inlier_ratio": inlier_ratio,
        })

    if not candidates:
        return None

    best = max(
        candidates,
        key=lambda item: (item["inliers"], item["inlier_ratio"], item["good_matches"]),
    )
    if (
        best["good_matches"] < MIN_GOOD_MATCHES
        or best["inliers"] < MIN_INLIERS
        or best["inlier_ratio"] < MIN_INLIER_RATIO
    ):
        return None
    return best
