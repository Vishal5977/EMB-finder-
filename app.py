import streamlit as st
import torch
import open_clip
import numpy as np
from PIL import Image, ImageDraw
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchAny, PointStruct, VectorParams
import streamlit.elements.image as st_image
from streamlit.elements.lib.image_utils import image_to_url as streamlit_image_to_url
from streamlit.elements.lib.layout_utils import LayoutConfig
from streamlit_drawable_canvas import st_canvas
from streamlit_cropper import st_cropper
import os
import pandas as pd
import re
import cv2
import json
import uuid
from datetime import datetime
   
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

QDRANT_PATH = r'C:\Users\Asus\embroidery-finder\qdrant_db'
DATABASE_CSV = r'C:\Users\Asus\embroidery-finder\design_database.csv'
MEMORY_CSV = r'C:\Users\Asus\embroidery-finder\match_memory.csv'
DESIGN_COLLECTION = "embroidery_designs"
MEMORY_COLLECTION = "embroidery_match_memory"
VECTOR_SIZE = 768
HIGH_CONFIDENCE_SCORE = 0.92
MEMORY_SCORE_THRESHOLD = 0.82

if not hasattr(st_image, "image_to_url"):
    def canvas_image_to_url(image, width, clamp, channels, output_format, image_id):
        return streamlit_image_to_url(
            image,
            LayoutConfig(width=width),
            clamp,
            channels,
            output_format,
            image_id,
        )

    st_image.image_to_url = canvas_image_to_url

design_df = pd.read_csv(DATABASE_CSV)

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

def display_design_code(design_name, file_name=""):
    design_text = str(design_name)
    file_text = str(file_name)
    design_numbers = re.findall(r'\b\d{3,5}\b', design_text)
    file_numbers = re.findall(r'\b\d{3,5}\b', file_text)

    if (
        re.search(r'\bDesign\s+0\d{3}\b', design_text, re.IGNORECASE)
        and file_numbers
        and file_numbers[0] not in design_numbers
    ):
        return file_numbers[0]

    return design_text

def build_number_to_designs(search_df):
    number_to_designs = {}
    for design_name in search_df['design_name'].astype(str).unique():
        numbers_in_name = re.findall(r'\d{3,5}', design_name)
        for num in numbers_in_name:
            if num not in number_to_designs:
                number_to_designs[num] = set()
            number_to_designs[num].add(design_name)
    return number_to_designs

def find_code_in_image(image_cv2, search_df):
    number_to_designs = build_number_to_designs(search_df)
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
        result_rows = search_df[search_df['design_name'].astype(str) == best_design]
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
    ensure_memory_collection(client)
    migrate_csv_memory(client)
    return client

def collection_exists(client, collection_name):
    try:
        client.get_collection(collection_name=collection_name)
        return True
    except Exception:
        return False

def ensure_memory_collection(client):
    if collection_exists(client, MEMORY_COLLECTION):
        return

    client.create_collection(
        collection_name=MEMORY_COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

def migrate_csv_memory(client):
    if not os.path.exists(MEMORY_CSV):
        return

    memory_df = pd.read_csv(MEMORY_CSV)
    if memory_df.empty or "vector" not in memory_df.columns:
        return

    existing, _ = client.scroll(
        collection_name=MEMORY_COLLECTION,
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    if existing:
        return

    points = []
    for _, row in memory_df.iterrows():
        try:
            vector = json.loads(row["vector"])
        except Exception:
            continue

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "saved_at": str(row.get("saved_at", "")),
                    "design_number": str(row.get("design_number", "")),
                    "designer": str(row.get("designer", "")).strip(),
                    "design_name": str(row.get("design_name", "")),
                    "source": "csv_migration",
                },
            )
        )

    if points:
        client.upsert(collection_name=MEMORY_COLLECTION, points=points)

def to_edges_3ch(pil_image, blur_kernel=9):
    img_array = np.array(pil_image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)

def auto_cleanup_embroidery(image):
    img_array = np.array(image.convert('RGB'))
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    color_mask = cv2.inRange(hsv, np.array([0, 35, 25]), np.array([180, 255, 255]))
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 35, 120)
    mask = cv2.bitwise_or(color_mask, cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)

    cleaned = np.full_like(img_array, 255)
    cleaned[mask > 0] = img_array[mask > 0]
    return Image.fromarray(cleaned)

def enhance_contrast(image):
    img_array = np.array(image.convert('RGB'))
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced)

def search_variants(image):
    return [
        ("normal", image),
        ("cleaned", auto_cleanup_embroidery(image)),
        ("contrast", enhance_contrast(image)),
        ("rotate_left", image.rotate(-8, expand=True, fillcolor=(255, 255, 255))),
        ("rotate_right", image.rotate(8, expand=True, fillcolor=(255, 255, 255))),
    ]

def query_vector_for_image(image, model, preprocess, device):
    edge_image = to_edges_3ch(image)
    image_tensor = preprocess(edge_image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()[0]

def designer_query_filter(selected_designers):
    if selected_designers:
        return Filter(
            must=[
                FieldCondition(
                    key="designer",
                    match=MatchAny(any=selected_designers),
                )
            ]
        )
    return None

def search_design(image, model, preprocess, device, client, selected_designers, top_k=35):
    query_filter = designer_query_filter(selected_designers)
    combined_results = []
    for _, variant_image in search_variants(image):
        query_vector = query_vector_for_image(variant_image, model, preprocess, device).tolist()
        combined_results.extend(
            client.query_points(
                collection_name=DESIGN_COLLECTION,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k
            ).points
        )
    return combined_results

def save_memory_match(image, model, preprocess, device, client, design_number, rows):
    first_row = rows.iloc[0]
    vector = query_vector_for_image(image, model, preprocess, device).tolist()
    client.upsert(
        collection_name=MEMORY_COLLECTION,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "saved_at": datetime.now().isoformat(timespec="seconds"),
                    "design_number": str(design_number).strip(),
                    "designer": str(first_row["designer"]).strip(),
                    "design_name": str(first_row["design_name"]),
                    "source": "review",
                },
            )
        ],
    )

def find_rows_by_design_number(design_number, search_df):
    design_number = str(design_number).strip()
    if not design_number:
        return pd.DataFrame(columns=search_df.columns)

    pattern = re.escape(design_number)
    matches = search_df[
        search_df["design_name"].astype(str).str.contains(pattern, case=False, na=False)
        | search_df["file_name"].astype(str).str.contains(pattern, case=False, na=False)
    ]
    if matches.empty:
        return matches

    first = matches.iloc[0]
    return search_df[
        (search_df["designer"].astype(str).str.strip() == str(first["designer"]).strip())
        & (search_df["design_name"].astype(str) == str(first["design_name"]))
    ]

def find_memory_matches(image, search_df, model, preprocess, device, client, selected_designers, threshold=MEMORY_SCORE_THRESHOLD):
    if not collection_exists(client, MEMORY_COLLECTION):
        return []

    query_filter = designer_query_filter(selected_designers)
    matches = []

    for variant_name, variant_image in search_variants(image):
        query_vector = query_vector_for_image(variant_image, model, preprocess, device).tolist()
        memory_points = client.query_points(
            collection_name=MEMORY_COLLECTION,
            query=query_vector,
            query_filter=query_filter,
            limit=5,
            score_threshold=threshold,
        ).points

        for point in memory_points:
            designer = str(point.payload.get("designer", "")).strip()
            design_name = str(point.payload.get("design_name", ""))
            rows = search_df[
                (search_df["designer"].astype(str).str.strip() == designer)
                & (search_df["design_name"].astype(str) == design_name)
            ]
            if rows.empty:
                continue

            matches.append({
                "score": point.score,
                "variant": variant_name,
                "design_number": str(point.payload.get("design_number", "")),
                "designer": designer,
                "design_name": design_name,
                "rows": rows,
            })

    best_by_design = {}
    for match in matches:
        key = (match["designer"], match["design_name"])
        if key not in best_by_design or match["score"] > best_by_design[key]["score"]:
            best_by_design[key] = match

    return sorted(best_by_design.values(), key=lambda item: item["score"], reverse=True)[:3]

def group_visual_results(results, search_df, max_groups=5):
    groups = {}
    for result in results:
        designer = str(result.payload.get("designer", "Unknown")).strip()
        design_name = str(result.payload["design_name"])
        key = (designer, design_name)

        if key not in groups:
            design_rows = search_df[
                (search_df["designer"].astype(str).str.strip() == designer)
                & (search_df["design_name"].astype(str) == design_name)
            ]
            groups[key] = {
                "designer": designer,
                "design_name": design_name,
                "best_score": result.score,
                "matched_files": [],
                "rows": design_rows,
            }

        groups[key]["best_score"] = max(groups[key]["best_score"], result.score)
        groups[key]["matched_files"].append({
            "file_name": result.payload["file_name"],
            "score": result.score,
        })

    ranked_groups = sorted(
        groups.values(),
        key=lambda group: (group["best_score"], len(group["matched_files"])),
        reverse=True,
    )
    return ranked_groups[:max_groups]

def resize_for_canvas(image, max_width=700):
    width, height = image.size
    if width <= max_width:
        return image, 1.0

    scale = max_width / width
    display_size = (max_width, int(height * scale))
    return image.resize(display_size), scale

def path_points(path_commands):
    points = []
    for command in path_commands:
        if not command:
            continue
        values = command[1:]
        if len(values) >= 2:
            points.append((float(values[-2]), float(values[-1])))
    return points

def build_manual_mask(canvas_json, display_size, original_size):
    if not canvas_json or not canvas_json.get("objects"):
        return None

    mask = Image.new("L", display_size, 0)
    draw = ImageDraw.Draw(mask)

    for item in canvas_json["objects"]:
        stroke_width = int(float(item.get("strokeWidth", 20)))
        if item.get("type") == "path":
            points = path_points(item.get("path", []))
            if len(points) >= 2:
                draw.line(points, fill=255, width=max(1, stroke_width), joint="curve")
                radius = max(1, stroke_width // 2)
                for x, y in points:
                    draw.ellipse(
                        (x - radius, y - radius, x + radius, y + radius),
                        fill=255,
                    )

    if not mask.getbbox():
        return None

    return mask.resize(original_size)

def apply_manual_mask(image, mask):
    background = Image.new("RGB", image.size, (255, 255, 255))
    background.paste(image.convert("RGB"), mask=mask)
    return background


st.title("Krishna Embroidery Finder")
st.subheader("Upload a customer image to find matching DST files")

with st.spinner("Loading AI model..."):
    model, preprocess, device = load_model()
    client = load_database()

st.success("AI Ready! Upload an image to search.")

designer_counts = (
    design_df["designer"]
    .astype(str)
    .str.strip()
    .value_counts()
    .sort_index(key=lambda index: index.str.lower())
)
designer_options = designer_counts.index.tolist()
selected_designers = st.multiselect(
    "Designer filter",
    designer_options,
    format_func=lambda designer: f"{designer} ({designer_counts[designer]})",
    help="Select one or more designers to search only those designs. Leave empty to search all designers.",
)

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

    search_image = cropped_image.convert('RGB')
    use_manual_mask = st.checkbox("Manual embroidery mask")
    if use_manual_mask:
        st.subheader("Step 3: Brush embroidery area")
        brush_size = st.slider("Brush size", 10, 80, 35, step=5)
        display_image, canvas_scale = resize_for_canvas(search_image)
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.25)",
            stroke_width=brush_size,
            stroke_color="rgba(255, 0, 0, 0.75)",
            background_image=display_image,
            update_streamlit=True,
            height=display_image.height,
            width=display_image.width,
            drawing_mode="freedraw",
            key="embroidery_mask_canvas",
        )
        mask = build_manual_mask(
            canvas_result.json_data if canvas_result else None,
            display_image.size,
            search_image.size,
        )
        if mask is not None:
            search_image = apply_manual_mask(search_image, mask)
            st.image(search_image, caption="Masked image used for search", width=300)
        else:
            st.caption("Brush over the embroidery area to use a manual mask.")

    if st.button("Find Matching Designs"):
        image = search_image.convert('RGB')
        img_array = np.array(image)
        img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        search_df = design_df
        if selected_designers:
            search_df = design_df[
                design_df["designer"].astype(str).str.strip().isin(selected_designers)
            ]

        st.caption(
            f"Searching {len(search_df)} designs"
            + (
                f" from {len(selected_designers)} selected designer(s)."
                if selected_designers
                else " from all designers."
            )
        )

        with st.spinner("Step 1: Checking for design code..."):
            code, code_rows = find_code_in_image(img_cv2, search_df)

        if code:
            st.success(f"Design Code Found: {code} - Exact Match!")
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
            with st.spinner("Step 2: Checking catalogue photos..."):
                catalogue_match = find_catalogue_match(image, selected_designers)

            if catalogue_match:
                matched_designer = catalogue_match["designer"]
                matched_design = catalogue_match["design_name"]
                catalogue_rows = search_df[
                    (search_df["designer"].astype(str) == matched_designer)
                    & (search_df["design_name"].astype(str) == matched_design)
                ]

                st.success(
                    f"Catalogue Match Found: {matched_design} "
                    f"({matched_designer}) - Exact Photo Match!"
                )
                st.caption(
                    f"Verified with {catalogue_match['inliers']} matching image points "
                    f"({catalogue_match['inlier_ratio'] * 100:.1f}% agreement)."
                )
                st.subheader(f"DST Files for {matched_design}:")

                for idx, row in catalogue_rows.iterrows():
                    with st.expander(f"{row['file_name']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            if os.path.exists(row["image_path"]):
                                st.image(row["image_path"], caption="Design Preview")
                        with col2:
                            st.write("**Design Code:**", matched_design)
                            st.write("**Designer:**", matched_designer)
                            st.write("**File:**", row["file_name"])
                            st.write("**DST File Location:**")
                            st.code(row["dst_path"])
            else:
                st.info("No catalogue photo matched. Searching by visual pattern instead...")

                with st.spinner("Checking saved match memory..."):
                    memory_matches = find_memory_matches(
                        image,
                        search_df,
                        model,
                        preprocess,
                        device,
                        selected_designers,
                    )

                if memory_matches:
                    st.subheader("Saved Match Memory")
                    for i, memory_match in enumerate(memory_matches):
                        memory_score = round(memory_match["score"] * 100, 2)
                        memory_rows = memory_match["rows"]
                        with st.expander(
                            f"Memory {i+1} - Design: {memory_match['design_number']} | "
                            f"Designer: {memory_match['designer']} | Similarity: {memory_score}%"
                        ):
                            st.write("**Design Code:**", memory_match["design_number"])
                            st.write("**Designer:**", memory_match["designer"])
                            st.write("**Memory Similarity:**", f"{memory_score}%")
                            for _, row in memory_rows.iterrows():
                                with st.container():
                                    st.markdown(f"**{row['file_name']}**")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if os.path.exists(row["image_path"]):
                                            st.image(row["image_path"], caption="Design Preview")
                                    with col2:
                                        st.write("**File:**", row["file_name"])
                                        st.write("**DST File Location:**")
                                        st.code(row["dst_path"])

                with st.spinner("Step 3: Multi-version visual similarity search..."):
                    results = search_design(
                        image,
                        model,
                        preprocess,
                        device,
                        client,
                        selected_designers,
                    )

                st.subheader("Top Matching Designs (Visual Search):")

                if not results:
                    st.warning("No visual matches found for the selected designer filter.")
                else:
                    st.caption(
                        "Searched normal, cleaned-background, contrast, and rotated versions, "
                        "then grouped the results by design."
                    )

                grouped_results = group_visual_results(results, search_df)

                for i, group in enumerate(grouped_results):
                    score = round(group["best_score"] * 100, 2)
                    design_name = group["design_name"]
                    designer = group["designer"]
                    rows = group["rows"]
                    sample_file = rows.iloc[0]["file_name"] if not rows.empty else ""
                    display_code = display_design_code(design_name, sample_file)
                    matched_files = sorted(
                        group["matched_files"],
                        key=lambda item: item["score"],
                        reverse=True,
                    )
                    matched_file_text = ", ".join(
                        f"{item['file_name']} ({round(item['score'] * 100, 2)}%)"
                        for item in matched_files[:3]
                    )

                    with st.expander(
                        f"Match {i+1} - Design: {display_code} | "
                        f"Designer: {designer} | Best Score: {score}% | "
                        f"Files: {len(rows)}"
                    ):
                        st.write("**Design Code:**", display_code)
                        st.write("**Designer:**", designer)
                        st.write("**Best Match Score:**", f"{score}%")
                        if matched_file_text:
                            st.write("**Closest matched files:**", matched_file_text)

                        for _, row in rows.iterrows():
                            file_score = next(
                                (
                                    item["score"]
                                    for item in matched_files
                                    if item["file_name"] == row["file_name"]
                                ),
                                None,
                            )
                            file_label = str(row["file_name"])
                            if file_score is not None:
                                file_label += f" - matched {round(file_score * 100, 2)}%"

                            with st.container():
                                st.markdown(f"**{file_label}**")
                                col1, col2 = st.columns(2)
                                with col1:
                                    if os.path.exists(row["image_path"]):
                                        st.image(row["image_path"], caption="Design Preview")
                                with col2:
                                    st.write("**File:**", row["file_name"])
                                    st.write("**DST File Location:**")
                                    st.code(row["dst_path"])

                st.subheader("Review / Teach Correct Design")
                with st.form("teach_correct_design_form"):
                    correct_design_number = st.text_input(
                        "Correct design number",
                        help="If the app missed the match, enter the correct design number here to save it for future searches.",
                    )
                    submitted = st.form_submit_button("Save Correct Match")

                if submitted:
                    corrected_rows = find_rows_by_design_number(correct_design_number, search_df)
                    if corrected_rows.empty:
                        st.error("I could not find that design number in the current designer filter/database.")
                    else:
                        save_memory_match(
                            image,
                            model,
                            preprocess,
                            device,
                            correct_design_number,
                            corrected_rows,
                        )
                        corrected_first = corrected_rows.iloc[0]
                        st.success(
                            f"Saved memory for design {correct_design_number} "
                            f"({corrected_first['designer']} / {corrected_first['design_name']})."
                        )
