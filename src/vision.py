from io import BytesIO
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

def _to_rgb_array(file_bytes):
    image = Image.open(BytesIO(file_bytes)).convert("RGB")
    return np.array(image)

def compare_images(previous_bytes, current_bytes):
    """
    Local computer-vision comparison.
    Returns an SSIM similarity score, changed-area percentage and an annotated image.
    """
    prev_rgb = _to_rgb_array(previous_bytes)
    curr_rgb = _to_rgb_array(current_bytes)

    h, w = prev_rgb.shape[:2]
    curr_rgb = cv2.resize(curr_rgb, (w, h), interpolation=cv2.INTER_AREA)

    prev_gray = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY)

    similarity, diff = ssim(prev_gray, curr_gray, full=True)
    diff_img = ((1.0 - diff) * 255).astype("uint8")

    _, thresh = cv2.threshold(
        diff_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    changed_pixels = int(np.count_nonzero(thresh))
    changed_pct = changed_pixels / float(thresh.size) * 100.0

    annotated = curr_rgb.copy()
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    minimum_area = max(150, int(w * h * 0.002))

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < minimum_area:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        boxes.append((x, y, bw, bh))
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (255, 185, 0), 3)

    return {
        "similarity": round(float(similarity) * 100.0, 1),
        "changed_area_pct": round(float(changed_pct), 2),
        "boxes": boxes,
        "annotated_rgb": annotated,
        "difference_mask": thresh,
    }
