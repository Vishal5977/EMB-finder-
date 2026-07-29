import cv2
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torchvision import transforms

DATABASE_CSV = r'C:\Users\Asus\embroidery-finder\design_database.csv'
CUSTOMER_IMAGE = r'C:\Users\Asus\embroidery-finder\test_customer.jpg'
BATCH_SIZE = 32

transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def letterbox(image):
    image = image.convert('RGB')
    width, height = image.size
    side = max(width, height)
    canvas = Image.new('RGB', (side, side), 'black')
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def edge_image(image, customer=False):
    array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    if customer:
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(gray, 80, 200)
    else:
        edges = cv2.Canny(gray, 50, 150)
    return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)
model = timm.create_model(
    'vit_small_patch14_dinov2.lvd142m',
    pretrained=True,
    num_classes=0,
    dynamic_img_size=True,
)
model.eval().to(device)


def encode(images):
    tensor = torch.stack([transform(image) for image in images]).to(device)
    with torch.no_grad():
        features = model(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu()


customer = Image.open(CUSTOMER_IMAGE).convert('RGB')
query_color = encode([letterbox(customer)])[0]
query_edge = encode([letterbox(edge_image(customer, customer=True))])[0]

df = pd.read_csv(DATABASE_CSV)
results = []
for start in range(0, len(df), BATCH_SIZE):
    batch = df.iloc[start:start + BATCH_SIZE]
    color_images = []
    edge_images = []
    rows = []
    for _, row in batch.iterrows():
        try:
            image = Image.open(row['image_path']).convert('RGB')
            color_images.append(letterbox(image))
            edge_images.append(letterbox(edge_image(image)))
            rows.append(row)
        except Exception as error:
            print('FAILED:', str(row['image_path']).encode('ascii', 'replace').decode(), error)
    if rows:
        color_features = encode(color_images)
        edge_features = encode(edge_images)
        color_scores = color_features @ query_color
        edge_scores = edge_features @ query_edge
        for row, color_score, edge_score in zip(rows, color_scores.tolist(), edge_scores.tolist()):
            combined = 0.35 * color_score + 0.65 * edge_score
            results.append((combined, edge_score, color_score, row['designer'], row['design_name'], row['file_name']))
    print(f'Processed {min(start + BATCH_SIZE, len(df))}/{len(df)}')

for label, score_index in [('COMBINED', 0), ('EDGE', 1), ('COLOR', 2)]:
    ranked = sorted(results, reverse=True, key=lambda item: item[score_index])
    print(f'\n{label} TOP 20')
    for rank, item in enumerate(ranked[:20], 1):
        combined, edge_score, color_score, designer, design_name, file_name = item
        print(rank, 'combined', round(combined * 100, 2), 'edge', round(edge_score * 100, 2), 'color', round(color_score * 100, 2), designer, design_name, file_name)
    print(f'\n{label} 2660 RESULTS')
    for rank, item in enumerate(ranked, 1):
        combined, edge_score, color_score, designer, design_name, file_name = item
        if '2660' in str(design_name):
            print(rank, 'combined', round(combined * 100, 2), 'edge', round(edge_score * 100, 2), 'color', round(color_score * 100, 2), designer, design_name, file_name)
