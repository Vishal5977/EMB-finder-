import torch
import open_clip
from PIL import Image
import cv2
import numpy as np

device = "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
model.eval()
model = model.to(device)

def get_embedding(pil_image):
    image_tensor = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    return features

def to_edges_blurred(pil_image, blur_kernel=1):
    img_array = np.array(pil_image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    if blur_kernel > 1:
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    else:
        blurred = gray
    edges = cv2.Canny(blurred, 50, 150)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)

# Test cases: (customer_image_path, correct_db_image_path, label)
TEST_CASES = [
    (
        r'C:\Users\Asus\embroidery-finder\testset\img1.jpg',
        r'C:\Users\Asus\embroidery-finder\dst_images\Bindu_Design 001_HBA528 FULL NECK.png',
        'Design 001 (Bindu)'
    ),
    (
        r'C:\Users\Asus\embroidery-finder\testset\img2.jpg',
        r'C:\Users\Asus\embroidery-finder\dst_images\Bindu_0572_front.png',
        '0572 (Bindu)'
    ),
]

BLUR_VALUES = [1, 5, 9, 15]

print("=" * 70)
print(f"{'Test Case':<25} {'Blur':>6} {'Score':>8}")
print("=" * 70)

for customer_path, db_path, label in TEST_CASES:
    customer = Image.open(customer_path).convert('RGB')
    db_image = Image.open(db_path).convert('RGB')

    for blur_k in BLUR_VALUES:
        c_edges = to_edges_blurred(customer, blur_kernel=blur_k)
        d_edges = to_edges_blurred(db_image, blur_kernel=blur_k)

        e1 = get_embedding(c_edges)
        e2 = get_embedding(d_edges)
        sim = torch.nn.functional.cosine_similarity(e1, e2).item()
        blur_label = "No blur" if blur_k == 1 else f"Blur {blur_k}"
        print(f"{label:<25} {blur_label:>8} {sim*100:>7.2f}%")

    print("-" * 70)

print("Done!")