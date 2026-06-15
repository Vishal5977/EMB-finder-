import streamlit as st
import torch
import open_clip
import numpy as np
from PIL import Image
from qdrant_client import QdrantClient
from streamlit_cropper import st_cropper
import os
import pandas as pd
import re
import cv2
   
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

QDRANT_PATH = r'C:\Users\Asus\embroidery-finder\qdrant_db'
DATABASE_CSV = r'C:\Users\Asus\embroidery-finder\design_database.csv'

design_df = pd.read_csv(DATABASE_CSV)

number_to_designs = {}
for design_name in design_df['design_name'].astype(str).unique():
    numbers_in_name = re.findall(r'\d{3,5}', design_name)
    for num in numbers_in_name:
        if num not in number_to_designs:
            number_to_designs[num] = set()
        number_to_designs[num].add(design_name)

COLOR_RANGES = {
    "pink_magenta": ([130, 40, 40], [175, 255, 255]),
    "white": ([0, 0, 120], [180, 80, 255]),
    "yellow": ([20, 50, 50], [35, 255, 255]),
    "cyan": ([85, 50, 50], [105, 255, 255]),
    "red": ([0, 50, 50], [10, 255, 255]),
    "red2": ([170, 50, 50], [180, 255, 255]),
    "green": ([40, 50, 50], [80, 255, 255]),
    "orange": ([10, 50, 50], [20, 255, 255]),
    "purple_blue": ([105, 50, 50], [130, 255, 255]),
}

def find_code_in_image(image_cv2):
    hsv = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2HSV)
    all_codes_found = {}

    for color_name, (lower, upper) in COLOR_RANGES.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        if cv2.countNonZero(mask) < 100:
            continue
        for config in ['--psm 11', '--psm 6']:
            text = pytesseract.image_to_string(mask, config=config)
            codes = re.findall(r'\b\d{3,5}\b', text)
            for code in codes:
                all_codes_found[code] = all_codes_found.get(code, 0) + 1

    valid_matches = []
    for ocr_num, freq in all_codes_found.items():
        if ocr_num in number_to_designs:
            for design_name in number_to_designs[ocr_num]:
                valid_matches.append((ocr_num, design_name, freq))

    if valid_matches:
        valid_matches.sort(key=lambda x: x[2], reverse=True)
        seen_designs = []
        for ocr_num, design_name, freq in valid_matches:
            if design_name not in seen_designs:
                seen_designs.append(design_name)

        best_design = seen_designs[0]
        result_rows = design_df[design_df['design_name'].astype(str) == best_design]
        return best_design, result_rows
    else:
        return None, None


@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    model.eval()
    model = model.to(device)
    return model, preprocess, device

@st.cache_resource
def load_database():
    client = QdrantClient(path=QDRANT_PATH)
    return client

def to_edges_3ch(pil_image):
    img_array = np.array(pil_image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)

def search_design(image, model, preprocess, device, client, top_k=5):
    edge_image = to_edges_3ch(image)
    image_tensor = preprocess(edge_image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        query_vector = features.cpu().numpy()[0].tolist()
    results = client.query_points(
        collection_name="embroidery_designs",
        query=query_vector,
        limit=top_k
    ).points
    return results

    # Group by (design_name, designer) - same code from different designers stays separate
    code_groups = {}
    for r in raw_results:
        key = (r.payload['design_name'], r.payload.get('designer', 'Unknown'))
        if key not in code_groups:
            code_groups[key] = []
        if len(code_groups[key]) < cap_per_code:
            code_groups[key].append(r)

    # Compute ranking score: (count, avg_score)
    ranked_codes = []
    for key, matches in code_groups.items():
        count = len(matches)
        avg_score = sum(m.score for m in matches) / count
        ranked_codes.append((key, count, avg_score, matches))

    # Sort by count desc, then avg_score desc
    ranked_codes.sort(key=lambda x: (x[1], x[2]), reverse=True)

    return ranked_codes[:top_k]


st.title("🧵 Krishna Embroidery Finder")
st.subheader("Upload a customer image to find matching DST files")

with st.spinner("Loading AI model..."):
    model, preprocess, device = load_model()
    client = load_database()

st.success("AI Ready! Upload an image to search.")

uploaded_file = st.file_uploader("Upload Customer Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')

    st.subheader("Step 1: Rotate (if needed)")
    rotation = st.slider("Rotate image (degrees)", -180, 180, 0, step=1)
    if rotation != 0:
        image = image.rotate(-rotation, expand=True, fillcolor=(0, 0, 0))

    st.subheader("Step 2: Crop to embroidery area")
    cropped_image = st_cropper(
        image,
        realtime_update=True,
        box_color='#FF0000',
        aspect_ratio=None
    )
    st.write(f"Cropped image size: {cropped_image.size}")
    st.subheader("Preview")
    st.image(cropped_image, caption="Cropped Image (this will be searched)", width=300)

    if st.button("🔍 Find Matching Designs"):
        image = cropped_image.convert('RGB')
        img_array = np.array(image)
        img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        with st.spinner("Step 1: Checking for design code..."):
            code, code_rows = find_code_in_image(img_cv2)

        if code:
            st.success(f"✅ Design Code Found: {code} — Exact Match!")
            st.subheader(f"DST Files for Design {code}:")

            for idx, row in code_rows.iterrows():
                with st.expander(f"{row['file_name']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        if os.path.exists(row['image_path']):
                            st.image(row['image_path'], caption="Design Preview")
                    with col2:
                        st.write("**Design Code:**", code)
                        st.write("**File:**", row['file_name'])
                        st.write("**DST File Location:**")
                        st.code(row['dst_path'])
        else:
            st.info("No design code detected. Searching by visual pattern instead...")

            with st.spinner("Step 2: Visual similarity search..."):
                results = search_design(image, model, preprocess, device, client)

            st.subheader("Top Matching Designs (Visual Search):")

            for i, result in enumerate(results):
                score = round(result.score * 100, 2)
                dst_path = result.payload['dst_path']
                design_name = result.payload['design_name']
                file_name = result.payload['file_name']
                image_path = result.payload['image_path']
                designer = result.payload.get('designer', 'Unknown')

                with st.expander(f"Match {i+1} — Code: {design_name} ({designer}) — {file_name} — Score: {score}%"):
                    col1, col2 = st.columns(2)
                    with col1:
                        if os.path.exists(image_path):
                            st.image(image_path, caption="Design Preview")
                    with col2:
                        st.write("**Design Code:**", design_name)
                        st.write("**Designer:**", designer)
                        st.write("**File:**", file_name)
                        st.write("**Match Score:**", f"{score}%")
                        st.write("**DST File Location:**")
                        st.code(dst_path)

