import re
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import cv2

from src.utils import ocr_image
from src.parsing.time_parsing import parse_time_fragment
from src.parsing.app_name_parsing import clean_app_name, is_valid_app_name

# ================================
# OCR PREPROCESSING
# ================================

def preprocess_for_ocr(image_path: str, light_text: bool = False) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.grayscale(img)

    contrast = 2.2 if light_text else 2.0
    brightness = 1.3 if light_text else 1.2
    threshold_val = 150 if light_text else 180

    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)

    img_np = np.array(img)
    _, img_np = cv2.threshold(img_np, threshold_val, 255, cv2.THRESH_BINARY_INV)

    return Image.fromarray(img_np)

# ================================
# PIXEL CLASSIFICATION
# ================================

HOURS = [f"{h if h!=0 else 12}am" if h < 12 else f"{h-12 if h>12 else 12}pm" for h in range(24)]

def classify_pixel(r, g, b):
    r, g, b = int(r), int(g), int(b)

    # 🥇 Most popular — Blue
    if r < 80 and 100 <= g <= 160 and b > 220:
        return "top1"

    # 🥈 Second most — Teal / Turquoise
    if r < 150 and g > 170 and 180 < b < 240:
        return "top2"

    # 🥉 Third most — Orange
    if r > 210 and 140 <= g <= 190 and b < 90:
        return "top3"

    # Other (dark gray segments)
    if 48 <= r <= 68 and 48 <= g <= 68 and 48 <= b <= 68 and abs(r-g) < 5 and abs(g-b) < 5:
        return "other"

    return None

def is_chart_bg(r, g, b):
    return 20 <= int(r) <= 45 and 20 <= int(g) <= 45 and 20 <= int(b) <= 50

def is_gridline_pixel(r, g, b):
    r, g, b = int(r), int(g), int(b)
    if abs(r - g) > 8 or abs(g - b) > 8:
        return False
    brightness = (r + g + b) / 3
    return 50 <= brightness <= 110

# ================================
# HOURLY CHART EXTRACTION
# ================================

def extract_hourly_chart(image_path: str) -> dict:
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    img_h, img_w = arr.shape[:2]

    def find_bar_regions(probe_x):
        regions = []
        in_region = False
        start = None
        for y in range(img_h // 4, img_h):
            has_bar = classify_pixel(*arr[y, probe_x]) is not None
            if has_bar and not in_region:
                start = y
                in_region = True
            elif not has_bar and in_region:
                if y - start > 50:
                    regions.append((start, y))
                in_region = False
        return regions

    vertical_probes = range(img_w // 3, 2 * img_w // 3, 40)
    best_regions = max(
        (find_bar_regions(x) for x in vertical_probes),
        key=len,
        default=[]
    )
    if not best_regions:
        return {}

    chart_top, chart_bottom = best_regions[-1]
    chart_bottom += 1

    gridlines = []
    for y in range(chart_top - 100, chart_bottom):
        if y < 0 or y >= img_h:
            continue
        row = arr[y]
        gray_count = sum(is_gridline_pixel(*px) for px in row)
        if gray_count > 0.4 * img_w:
            gridlines.append(y)

    collapsed = []
    if gridlines:
        group = [gridlines[0]]
        for y in gridlines[1:]:
            if y - group[-1] <= 2:
                group.append(y)
            else:
                collapsed.append(int(sum(group)/len(group)))
                group = [y]
        collapsed.append(int(sum(group)/len(group)))
    collapsed.sort()

    chart_top_line, chart_bottom_line = (
        (collapsed[0], collapsed[-1])
        if len(collapsed) >= 2
        else (chart_top, chart_bottom)
    )
    ymax = chart_bottom_line - chart_top_line

    def find_vertical_axis(search_from_left=True):
        x_range = range(img_w) if search_from_left else range(img_w - 1, -1, -1)
        for x in x_range:
            col = arr[chart_top_line:chart_bottom_line, x]
            count = sum(is_gridline_pixel(*px) for px in col)
            if count > 0.35 * (chart_bottom_line - chart_top_line):
                return x
        return None

    chart_left = find_vertical_axis(True)
    chart_right = find_vertical_axis(False)

    if chart_left is None or chart_right is None or chart_right <= chart_left:
        return {}

    total_width = chart_right - chart_left
    slot_width = total_width / 24.0

    slot_centers = []
    for i in range(24):
        start = chart_left + slot_width * i
        end   = chart_left + slot_width * (i+1)
        center = (start + end) / 2
        slot_centers.append(center)

    bar_segments = []
    in_bar = False
    seg_start = None

    for x in range(chart_left, chart_right):
        col = arr[chart_top_line:chart_bottom_line, x]
        bar_pixel_count = sum(classify_pixel(*px) is not None for px in col)
        has_bar = bar_pixel_count > 5

        if has_bar and not in_bar:
            seg_start = x
            in_bar = True
        elif not has_bar and in_bar:
            if x - seg_start >= 6:
                bar_segments.append((seg_start, x - 1))
            in_bar = False

    if in_bar and chart_right - seg_start >= 6:
        bar_segments.append((seg_start, chart_right - 1))

    result = {
        hour: {
            "overall": 0,
            "top1": 0,
            "top2": 0,
            "top3": 0,
            "other": 0
        }
        for hour in HOURS
    }


    result['ymax_pixels'] = ymax

    for x1, x2 in bar_segments:
        bar_center = (x1 + x2) / 2.0

        distances = [abs(bar_center - c) for c in slot_centers]
        slot_idx = distances.index(min(distances))
        hour = HOURS[slot_idx]

        best = {
            "overall": 0,
            "top1": 0,
            "top2": 0,
            "top3": 0,
            "other": 0
        }

        for x in range(x1, x2+1):
            col = arr[chart_top_line:chart_bottom_line, x]
            cats = [classify_pixel(*px) for px in col]
            bar_rows = [y for y,c in enumerate(cats) if c is not None]
            if len(bar_rows) < 2:
                continue

            bar_top = chart_top_line + min(bar_rows)
            bar_height = chart_bottom_line - bar_top

            if bar_height < 1:
                continue

            if bar_height > best['overall']:
                best['overall'] = bar_height
                best.update({
                    cat: sum(1 for c in cats if c == cat)
                    for cat in ["top1", "top2", "top3", "other"]
                })

        result[hour] = best

    return result


# ================================
# IOS SCREEN TIME PROCESSING
# ================================

def process_ios_overall_screenshot(image_path: str) -> dict:
    # --- OCR passes ---
    normal_text = ocr_image(preprocess_for_ocr(image_path))
    light_text = ocr_image(preprocess_for_ocr(image_path, light_text=True))
    lines = [l.strip() for l in (normal_text + "\n" + light_text).split("\n") if l.strip()]

    result = {
        "date": None,
        "is_yesterday": False,
        "total_time": "0h 0m",
        "categories": [],
        "top_apps": [],
        "ymax_pixels": None,
        "hourly_usage": {}
    }

    # --- Date detection ---
    for line in lines:
        if "yesterday" in line.lower():
            result["is_yesterday"] = True
            result["date"] = line.split(",")[-1].strip() if "," in line else line.strip()
            break

    # --- Total screen time ---
    in_total_section = False
    for line in lines:
        if "screen time" in line.lower():
            in_total_section = True
            continue
        if in_total_section:
            parsed = parse_time_fragment(line)
            if parsed:
                h,m = parsed
                if h+m > 0:
                    result["total_time"] = f"{h}h {m}m"
                    break

    # --- Category times ---
    for i,line in enumerate(lines):
        if any(cat in line.lower() for cat in ["social", "entertainment", "games"]):
            names = line.split()
            if i+1 < len(lines):
                times_line = lines[i+1]
                time_matches = re.findall(r"(\d+h\s*\d*m|\d+h|\d*m)", times_line)
                if time_matches and len(time_matches) == len(names):
                    for name, tstr in zip(names, time_matches):
                        h,m = parse_time_fragment(tstr)
                        result["categories"].append({"name":name,"time":f"{h}h {m}m"})
            break

    # --- Top apps ---
    top_apps = []
    all_app_entries = {}
    for i,line in enumerate(lines):
        if "most used" in line.lower():
            for j in range(i+1, min(i+11,len(lines))):
                candidate = lines[j].strip()
                candidate = re.sub(r'^[a-zA-Z&\d]{1,2}[\.\s]\s*', '', candidate)
                candidate = re.sub(r"['\-,\.]+$","",candidate)
                candidate = re.sub(r"^['\"]|['\"]$","",candidate)
                candidate = re.sub(r'\s+\d+$','',candidate)
                candidate = candidate.strip()
                name = clean_app_name(candidate)
                if not is_valid_app_name(name) or len(name)<3:
                    continue
                normalized = re.sub(r'\W+$','',name).strip()
                if len(normalized)<3:
                    continue
                all_app_entries.setdefault(normalized,[]).append(j)

    # Match apps with times
    time_map = {i:f"{h}h {m}m" for i,line in enumerate(lines) if (parsed:=parse_time_fragment(line)) and sum(parsed)>0 for h,m in [parsed]}
    for app_name,line_indices in all_app_entries.items():
        if len(top_apps)>=3:
            break
        app_time = "0h 0m"
        for idx in reversed(line_indices):
            for offset in range(1,4):
                check_idx = idx+offset
                if check_idx in time_map:
                    app_time = time_map.pop(check_idx)
                    break
            if app_time!="0h 0m":
                break
        top_apps.append({"name":app_name,"time":app_time})

    top_apps.sort(key=lambda x: sum(parse_time_fragment(x["time"])), reverse=True)
    result["top_apps"] = top_apps[:3]

    # --- Hourly chart ---
    hourly_raw = extract_hourly_chart(image_path)
    result["ymax_pixels"] = hourly_raw.pop("ymax_pixels", None)

    # Build dynamic color → category mapping
    color_to_category = {}

    if len(result["categories"]) >= 1:
        color_to_category["top1"] = result["categories"][0]["name"].lower()

    if len(result["categories"]) >= 2:
        color_to_category["top2"] = result["categories"][1]["name"].lower()

    if len(result["categories"]) >= 3:
        color_to_category["top3"] = result["categories"][2]["name"].lower()

    hourly_cleaned = {}

    for hour in HOURS:
        raw = hourly_raw.get(hour, {})

        hour_data = {
            "overall": raw.get("overall", 0),
            "social": 0,
            "entertainment": 0
        }

        for color_key, category_name in color_to_category.items():
            if category_name == "social":
                hour_data["social"] = raw.get(color_key, 0)

            if category_name == "entertainment":
                hour_data["entertainment"] = raw.get(color_key, 0)

        hourly_cleaned[hour] = hour_data

    result["hourly_usage"] = hourly_cleaned


    return result
