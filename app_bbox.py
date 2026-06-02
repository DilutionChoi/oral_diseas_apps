import io
import streamlit as st
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).parent
IMAGE_DIR = BASE_DIR / "test_bbox" / "images"
LABEL_DIR = BASE_DIR / "test_bbox" / "labels"
MODEL_PATH = BASE_DIR / "models" / "bbox_best.pt"

CLASS_NAMES = ["cavity", "normal"]
BOX_COLOR = (80, 210, 80)
DISPLAY_WIDTH = 640  # 브라우저 전송 크기 제한

st.set_page_config(layout="wide", page_title="BBox Viewer")


@st.cache_resource
def load_model():
    from ultralytics import YOLO
    return YOLO(str(MODEL_PATH))


@st.cache_data
def get_image_list():
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(IMAGE_DIR.glob(ext))
    return sorted(f.name for f in files)



def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    return (
        (cx - w / 2) * img_w,
        (cy - h / 2) * img_h,
        (cx + w / 2) * img_w,
        (cy + h / 2) * img_h,
    )


@st.cache_data
def render_pair(fname: str) -> tuple[bytes, bytes]:
    """이미지 로드 → GT/예측 bbox 그리기 → JPEG 바이트 반환. 결과 캐싱."""
    image = Image.open(IMAGE_DIR / fname).convert("RGB")
    w, h = image.size

    # display 크기로 리사이즈 (bbox 좌표도 같이 스케일)
    scale = min(1.0, DISPLAY_WIDTH / w)
    dw, dh = int(w * scale), int(h * scale)
    image = image.resize((dw, dh), Image.LANCZOS)

    # GT boxes
    label_path = LABEL_DIR / f"{Path(fname).stem}.txt"
    gt_img = image.copy()
    draw = ImageDraw.Draw(gt_img)
    if label_path.exists():
        for line in label_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            x1, y1, x2, y2 = yolo_to_xyxy(
                float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]),
                dw, dh,
            )
            draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=4)

    # Prediction boxes
    model = load_model()
    results = model(str(IMAGE_DIR / fname), verbose=False)[0]
    pred_img = image.copy()
    draw = ImageDraw.Draw(pred_img)
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:
        font = ImageFont.load_default()
    for box in results.boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [v * scale for v in box.xyxy[0].tolist()]
        draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=4)
        draw.text((x1 + 2, max(y1 - 16, 2)), f"{conf:.2f}", fill=BOX_COLOR, font=font)

    def to_jpeg(img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    return to_jpeg(gt_img), to_jpeg(pred_img)


# ── Page ─────────────────────────────────────────────────────────────────────
st.title("Gingivitis Detection — BBox")

col_ds, col_res = st.columns(2, gap="large")

with col_ds:
    st.markdown("""
**데이터셋** — [Intraoral Gingivitis Dataset](https://data.mendeley.com/datasets/3253gj88rr/1)

| 분할 | 이미지 수 | (background) |
|------|----------:|:-------------|
| Train | 732 | 100장 |
| Val | 182 | 20장 |
| Test | 182 | 28장 |
""")

with col_res:
    st.markdown("**모델 성능 (Test set)**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", "0.605")
    m2.metric("Recall", "0.797")
    m3.metric("mAP50", "0.719")
    m4.metric("mAP50-95", "0.405")

st.divider()

all_images = get_image_list()
st.caption(f"총 {len(all_images)}개  |  첫 실행 시 추론이 순차적으로 진행됩니다.")

h_gt, h_pred = st.columns(2)
h_gt.markdown("#### Ground Truth")
h_pred.markdown("#### Model Prediction")
st.divider()

for fname in all_images:
    gt_bytes, pred_bytes = render_pair(fname)
    col_gt, col_pred = st.columns(2)
    with col_gt:
        st.image(gt_bytes, use_container_width=True)
    with col_pred:
        st.image(pred_bytes, use_container_width=True)
    st.divider()
