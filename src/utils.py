import pytesseract
from PIL import Image
import cv2
import numpy as np
import re

# -----------------------
# OCR helper
# -----------------------
def ocr_image(image_or_path):
    if isinstance(image_or_path, str):
        image = Image.open(image_or_path)
    else:
        image = image_or_path  # already a PIL Image
    return pytesseract.image_to_string(image)


# -----------------------
# Extract overall info + hourly bar chart
# -----------------------
def extract_overall_info_and_bars(image_path):
    """
    Takes a single overall screenshot and returns:
    - date
    - total screen time
    - hourly breakdown (minutes per hour, 24 items)
    """
    # -----------------
    # 1️⃣ OCR text for date + total time
    # -----------------
    text = ocr_image(image_path)
    date_match = re.search(r"Yesterday,\s*(.+)", text)
    date = date_match.group(1) if date_match else ""
    time_match = re.search(r"(\d+)h\s*(\d+)m", text)
    total_time = f"{time_match.group(1)}h {time_match.group(2)}m" if time_match else ""

    # -----------------
    # 2️⃣ Load image for bar chart analysis
    # -----------------
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -----------------
    # 3️⃣ Crop right side for axis labels (0m → 60m)
    # -----------------
    right_crop = gray[:, int(img.shape[1]*0.85):]  # right 15%

    # Use pytesseract to get bounding boxes
    boxes = pytesseract.image_to_boxes(right_crop)  # char-level boxes
    h_crop = right_crop.shape[0]

    y_0m, y_60m = None, None
    labels = {}
    # Build a dictionary of lines for OCR
    for line in boxes.splitlines():
        parts = line.split(' ')
        char, x1, y1, x2, y2 = parts[0], int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
        # Tesseract coords: origin is bottom-left, OpenCV: origin is top-left
        y1 = h_crop - y1
        y2 = h_crop - y2
        y_mid = (y1 + y2) // 2

        if char in '0m':
            labels.setdefault("0m", []).append(y_mid)
        elif char in '6':
            labels.setdefault("60m", []).append(y_mid)

    # Use median y position for more robustness
    if "0m" in labels:
        y_0m = int(np.median(labels["0m"]))
    if "60m" in labels:
        y_60m = int(np.median(labels["60m"]))

    # Fallback if OCR fails
    if y_0m is None or y_60m is None:
        y_0m, y_60m = img.shape[0]-50, 50

    # -----------------
    # 4️⃣ Crop bar chart region (from 60m → 0m)
    # -----------------
    chart = gray[y_60m:y_0m, :]
    chart_height = y_0m - y_60m

    # -----------------
    # 5️⃣ Threshold to detect bars
    # -----------------
    _, thresh = cv2.threshold(chart, 200, 255, cv2.THRESH_BINARY_INV)

    # -----------------
    # 6️⃣ Split chart horizontally into 24 hourly bins
    # -----------------
    hour_width = chart.shape[1] / 24
    hourly_breakdown = []

    for i in range(24):
        x_start = int(i * hour_width)
        x_end = int((i + 1) * hour_width)
        col = thresh[:, x_start:x_end]

        # Topmost white pixel = top of the bar
        white_pixels = np.where(col > 0)
        if len(white_pixels[0]) == 0:
            bar_height = 0
        else:
            bar_top = white_pixels[0].min()
            bar_height = chart.shape[0] - bar_top  # distance from 0m line

        # Convert pixels → minutes
        minutes = (bar_height / chart_height) * 60
        hourly_breakdown.append(round(minutes))

    return {
        "date": date,
        "total_time": total_time,
        "hourly_breakdown": hourly_breakdown
    }
