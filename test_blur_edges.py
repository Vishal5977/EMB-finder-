import torch
import open_clip
from PIL import Image
import cv2
import numpy as np

device = "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
model.eval()
model = model.to(device)

CUSTOMER_IMG = r'C:\Users\Asus\embroidery-finder\testset\img2.jpg'
DB_IMG = r'C:\Users\Asus\embroidery-finder\dst_images\Bindu_0572_front.png'

def get_embedding(pil_image):
    image_tensor = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    return features

def to_edges_blurred(pil_image, blur_kernel=1, canny_low=50, canny_high=150):
    img_array = np.array(pil_image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    if blur_kernel > 1:
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    else:
        blurred = gray
    edges = cv2.Canny(blurred, canny_low, canny_high)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)

customer = Image.open(CUSTOMER_IMG).convert('RGB')
db_image = Image.open(DB_IMG).convert('RGB')

print("=" * 60)
print("Testing different blur amounts before edge detection")
print("=" * 60)

best_score = 0
best_blur = None

for blur_k in [9]:
    c_edges = to_edges_blurred(customer, blur_kernel=blur_k)
    d_edges = to_edges_blurred(db_image, blur_kernel=blur_k)

    e1 = get_embedding(c_edges)
    e2 = get_embedding(d_edges)
    sim = torch.nn.functional.cosine_similarity(e1, e2).item()
    label = "No blur" if blur_k == 1 else f"Blur kernel {blur_k}"
    print(f"{label:20s} -> Similarity: {sim*100:.2f}%")

    if sim > best_score:
        best_score = sim
        best_blur = blur_k

    if blur_k in [5]:
        c_edges.save(rf'C:\Users\Asus\embroidery-finder\edge_customer_blur{blur_k}.png')

print("=" * 60)
print(f"BEST: blur_kernel={best_blur}, score={best_score*100:.2f}%")
print("\nSaved edge_customer_blur15.png and edge_customer_blur31.png for inspection")