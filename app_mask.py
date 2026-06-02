import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="Dental Mask Segmentation",
    page_icon="🦷",
    layout="wide"
)

CLASS_NAMES = {0: "dental_calculus", 1: "dental_caries", 2: "gingivitis"}
CLASS_COLORS = {
    0: (255, 56, 56),
    1: (255, 157, 151),
    2: (255, 112, 31),
}

MODEL_PATH = r"models\mask_best.pt"
DATA_ROOT  = r"test_mask"
IMAGES_DIR = os.path.join(DATA_ROOT, "images")
LABELS_DIR = os.path.join(DATA_ROOT, "labels")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@st.cache_resource
def load_model():
    try:
        return YOLO(MODEL_PATH)
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        return None


def load_image_paths():
    paths = []
    for fname in sorted(os.listdir(IMAGES_DIR)):
        if os.path.splitext(fname)[1].lower() in IMAGE_EXTS:
            paths.append(os.path.join(IMAGES_DIR, fname))
    return paths


def find_label(image_path: str) -> str | None:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    path = os.path.join(LABELS_DIR, stem + ".txt")
    return path if os.path.exists(path) else None


def draw_gt_polygons(image_rgb: np.ndarray, label_path: str) -> np.ndarray:
    h, w   = image_rgb.shape[:2]
    result = image_rgb.copy()

    with open(label_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) < 7:
            continue
        cls_id = int(parts[0])
        coords = list(map(float, parts[1:]))
        pts = np.array(
            [[int(coords[i] * w), int(coords[i + 1] * h)] for i in range(0, len(coords) - 1, 2)],
            dtype=np.int32,
        )
        color   = CLASS_COLORS.get(cls_id, (255, 255, 0))
        overlay = result.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.35, result, 0.65, 0, result)
        cv2.polylines(result, [pts], isClosed=True, color=color, thickness=2)

    return result


def draw_pred_polygons(image_rgb: np.ndarray, result) -> np.ndarray:
    h, w   = image_rgb.shape[:2]
    canvas = image_rgb.copy()

    if result.masks is None:
        return canvas

    for mask, box in zip(result.masks.data, result.boxes):
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        color  = CLASS_COLORS.get(cls_id, (255, 255, 0))

        mask_np = mask.cpu().numpy()
        mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
        pts_list, _ = cv2.findContours(
            (mask_resized > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        overlay = canvas.copy()
        cv2.fillPoly(overlay, pts_list, color)
        cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)
        cv2.polylines(canvas, pts_list, isClosed=True, color=color, thickness=2)

        if pts_list:
            all_pts = np.concatenate(pts_list)
            cx, cy  = all_pts.mean(axis=0)[0].astype(int)
            label   = f"{conf:.2f}"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(canvas, (cx - 2, cy - th - 4), (cx + tw + 2, cy + bl), color, -1)
            cv2.putText(canvas, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


def render_info():
    swatches = " &nbsp;&nbsp;&nbsp; ".join(
        f'<span style="display:inline-block;width:16px;height:16px;border-radius:3px;'
        f'background:rgb{color};vertical-align:middle;margin-right:6px;"></span>'
        f'<span style="color:#ffffff;vertical-align:middle;font-weight:600;">{name}</span>'
        for (_, color), name in zip(CLASS_COLORS.items(), CLASS_NAMES.values())
    )
    st.markdown(
        f'<div style="background:#2c2c2c;padding:10px 16px;border-radius:8px;font-size:15px;margin-bottom:8px;">'
        f'{swatches}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "- **데이터셋**: [Oral screening](https://data.mendeley.com/datasets/3253gj88rr/1)\n"
        "- **모델**: YOLO11s-seg"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**데이터 Split**")
        st.table({"Split": ["train", "val", "test"], "이미지 수": ["1,823", "287", "287"]})
    with col2:
        st.markdown("**클래스별 분포**")
        st.table({
            "Class":           ["dental_calculus", "dental_caries", "gingivitis"],
            "train":           ["359", "906", "975"],
            "val":             ["57",  "186", "131"],
            "test":            ["56",  "182", "131"],
        })

    with st.expander("성능 지표 (Box / Mask)", expanded=False):
        st.markdown("**Box**")
        st.table({
            "Class":           ["all", "dental_calculus", "dental_caries", "gingivitis"],
            "P":               [0.436, 0.309, 0.596, 0.404],
            "R":               [0.465, 0.338, 0.649, 0.409],
            "mAP50":           [0.435, 0.261, 0.661, 0.385],
            "mAP50-95":        [0.195, 0.117, 0.298, 0.170],
        })
        st.markdown("**Mask**")
        st.table({
            "Class":           ["all", "dental_calculus", "dental_caries", "gingivitis"],
            "P":               [0.467, 0.327, 0.659, 0.414],
            "R":               [0.372, 0.255, 0.574, 0.289],
            "mAP50":           [0.371, 0.179, 0.629, 0.305],
            "mAP50-95":        [0.134, 0.054, 0.240, 0.108],
        })

    st.markdown("---")


def main():
    st.title("🦷 Dental Mask Segmentation")
    st.markdown("### Ground Truth vs Model Prediction")
    render_info()
    st.markdown("### 결과")

    model = load_model()
    if model is None:
        return

    image_paths = load_image_paths()
    if not image_paths:
        st.warning(f"`{IMAGES_DIR}` 폴더에 이미지가 없습니다.")
        return

    progress = st.progress(0, text="이미지 처리 중...")

    for idx, img_path in enumerate(image_paths):
        fname    = os.path.basename(img_path)
        image    = Image.open(img_path).convert("RGB")
        image_np = np.array(image)

        results    = model.predict(source=image, conf=0.25, verbose=False)
        pred_image = draw_pred_polygons(image_np, results[0])

        label_path = find_label(img_path)
        gt_image   = draw_gt_polygons(image_np, label_path) if label_path else image_np

        st.markdown(f"---\n#### {idx + 1}. `{fname}`")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Ground Truth")
            st.image(gt_image, use_container_width=True)
            if label_path is None:
                st.caption("라벨 파일 없음")

        with col2:
            st.subheader("Model Prediction")
            st.image(pred_image, use_container_width=True)

        progress.progress((idx + 1) / len(image_paths), text=f"{idx + 1}/{len(image_paths)} 처리 중...")

    progress.empty()
    st.success(f"총 {len(image_paths)}개 이미지 처리 완료!")


if __name__ == "__main__":
    main()
