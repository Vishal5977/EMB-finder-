import cv2
import numpy as np
import pandas as pd
from PIL import Image

DATABASE_CSV = r'C:\Users\Asus\embroidery-finder\design_database.csv'
CUSTOMER_IMAGE = r'C:\Users\Asus\embroidery-finder\test_customer.jpg'
MAX_SIDE = 1200


def resize_for_features(gray):
    height, width = gray.shape
    scale = min(1.0, MAX_SIDE / max(width, height))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return gray


def customer_edge(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = cv2.GaussianBlur(image, (7, 7), 0)
    return cv2.Canny(image, 80, 200)


def database_edge(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = resize_for_features(image)
    return cv2.Canny(image, 50, 150)


sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.02, edgeThreshold=15)
matcher = cv2.BFMatcher(cv2.NORM_L2)
query = resize_for_features(customer_edge(CUSTOMER_IMAGE))
query_points, query_descriptors = sift.detectAndCompute(query, None)
print('Query keypoints:', len(query_points))

df = pd.read_csv(DATABASE_CSV)
results = []
for position, row in df.iterrows():
    try:
        candidate = database_edge(row['image_path'])
        candidate_points, candidate_descriptors = sift.detectAndCompute(candidate, None)
        if query_descriptors is None or candidate_descriptors is None or len(candidate_descriptors) < 2:
            continue
        pairs = matcher.knnMatch(query_descriptors, candidate_descriptors, k=2)
        good = [first for first, second in pairs if first.distance < 0.72 * second.distance]
        inliers = 0
        if len(good) >= 6:
            source = np.float32([query_points[match.queryIdx].pt for match in good]).reshape(-1, 1, 2)
            target = np.float32([candidate_points[match.trainIdx].pt for match in good]).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(source, target, cv2.RANSAC, 8.0)
            if mask is not None:
                inliers = int(mask.sum())
        score = inliers * 3 + len(good)
        results.append((score, inliers, len(good), row['designer'], row['design_name'], row['file_name']))
    except Exception as error:
        print('FAILED:', row['image_path'], error)
    if (position + 1) % 100 == 0:
        print(f'Processed {position + 1}/{len(df)}')

results.sort(reverse=True, key=lambda item: (item[0], item[1], item[2]))
print('\nTOP 30')
for rank, item in enumerate(results[:30], 1):
    score, inliers, good, designer, design_name, file_name = item
    print(rank, 'score', score, 'inliers', inliers, 'good', good, designer, design_name, file_name)

print('\nALL 2660 RESULTS')
for rank, item in enumerate(results, 1):
    score, inliers, good, designer, design_name, file_name = item
    if '2660' in str(design_name):
        print(rank, 'score', score, 'inliers', inliers, 'good', good, designer, design_name, file_name)

