import re
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import cv2

from src.utils import ocr_image
from src.parsing.time_parsing import parse_time_fragment
from src.parsing.app_name_parsing import clean_app_name, is_valid_app_name

def preprocess_for_ocr(image_path: str) -> Image.Image:
    """Enhance contrast and brightness, threshold lightly colored text for OCR."""
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.grayscale(img)
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = ImageEnhance.Brightness(img).enhance(1.2)
    img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    img_np = np.array(img)
    _, img_np = cv2.threshold(img_np, 180, 255, cv2.THRESH_BINARY_INV)
    return Image.fromarray(img_np)

def preprocess_for_light_text(image_path: str) -> Image.Image:
    """Specifically optimized for very light gray text like the times."""
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.grayscale(img)
    img = ImageEnhance.Contrast(img).enhance(2.2)
    img = ImageEnhance.Brightness(img).enhance(1.3)
    img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    img_np = np.array(img)
    _, img_np = cv2.threshold(img_np, 150, 255, cv2.THRESH_BINARY_INV)
    return Image.fromarray(img_np)

HOURS = [
    '12am','1am','2am','3am','4am','5am','6am','7am','8am','9am','10am','11am',
    '12pm','1pm','2pm','3pm','4pm','5pm','6pm','7pm','8pm','9pm','10pm','11pm'
]

def classify_pixel(r, g, b):
    """
    Return color key for a bar pixel, or None if not a bar pixel.
    Colors: blue, teal, orange, gray.
    """
    r, g, b = int(r), int(g), int(b)
    if r < 80 and 100 <= g <= 160 and b > 220:
        return 'blue'
    if r < 150 and g > 170 and 180 < b < 240:
        return 'teal'
    if r > 210 and 140 <= g <= 190 and b < 90:
        return 'orange'
    if 48 <= r <= 68 and 48 <= g <= 68 and 48 <= b <= 68 and abs(r-g) < 5 and abs(g-b) < 5:
        return 'gray'
    return None

def is_chart_bg(r, g, b):
    return 20 <= int(r) <= 45 and 20 <= int(g) <= 45 and 20 <= int(b) <= 50

def extract_hourly_chart(image_path: str, color_to_category: dict) -> dict:
    """Detect hourly bars and classify by dynamic color-to-category mapping."""
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    img_h, img_w = arr.shape[:2]

    # Find vertical chart regions
    def find_bar_regions(probe_x):
        regions = []; in_region = False; start = None
        for y in range(img_h // 4, img_h):
            has_bar = classify_pixel(*arr[y, probe_x]) is not None
            if has_bar and not in_region:
                start = y; in_region = True
            elif not has_bar and in_region:
                if y - start > 50: regions.append((start, y))
                in_region = False
        return regions

    best_regions = []; best_probe_x = img_w // 2
    for probe_x in range(img_w // 3, 2 * img_w // 3, 40):
        regions = find_bar_regions(probe_x)
        if len(regions) > len(best_regions):
            best_regions = regions; best_probe_x = probe_x
    if not best_regions:
        return {}

    chart_start, chart_end = best_regions[-1]
    chart_bottom = chart_end + 1

    # Chart horizontal boundaries
    y_probe = chart_bottom - 5
    chart_left = next((x for x in range(img_w) if is_chart_bg(*arr[y_probe, x])), None)
    chart_right = next((x for x in range(img_w-1, -1, -1) if is_chart_bg(*arr[y_probe, x])), None)
    if chart_left is None or chart_right is None:
        return {}

    # Chart top
    chart_mid_x = (chart_left + chart_right) // 2
    first_bar_y = next(
        (y for y in range(chart_start - 50, chart_bottom)
         if 0 <= y < img_h and classify_pixel(*arr[y, chart_mid_x]) is not None),
        chart_start
    )
    chart_top = first_bar_y
    for y in range(first_bar_y, max(0, first_bar_y - 200), -1):
        r, g, b = [int(v) for v in arr[y, chart_mid_x]]
        if classify_pixel(r, g, b) is None and not (48 <= r <= 70 and abs(r-g) < 10 and abs(g-b) < 10):
            chart_top = y + 1; break

    corner_padding = 8
    ymax = (chart_bottom - chart_top) - corner_padding

    # Detect horizontal bars
    bar_segments = []
    in_bar = False; seg_start = None
    for x in range(chart_left, chart_right):
        col = arr[chart_top:chart_bottom, x]
        has_bar = any(classify_pixel(*px) is not None for px in col)
        if has_bar and not in_bar:
            seg_start = x; in_bar = True
        elif not has_bar and in_bar:
            if x - seg_start >= 10: bar_segments.append((seg_start, x - 1))
            in_bar = False
    if in_bar and chart_right - seg_start >= 10:
        bar_segments.append((seg_start, chart_right - 1))

    # Hour mapping
    if len(bar_segments) >= 2:
        centers = [(x1 + x2)/2 for x1, x2 in bar_segments]
        spacings = [centers[i+1]-centers[i] for i in range(len(centers)-1)]
        bar_spacing = sum(spacings)/len(spacings)
    else:
        bar_spacing = (chart_right - chart_left)/24

    result = {'ymax_pixels': ymax}
    for hour in HOURS:
        result[hour] = {cat:0 for cat in color_to_category.values()}

    if not bar_segments:
        return result

    first_center = (bar_segments[0][0] + bar_segments[0][1])/2
    approx_hour = round((first_center - chart_left)/((chart_right - chart_left)/24))
    hour_0_x = first_center - approx_hour*bar_spacing

    for x1, x2 in bar_segments:
        bar_center = (x1 + x2)/2
        slot_idx = max(0, min(23, round((bar_center - hour_0_x)/bar_spacing)))
        hour = HOURS[slot_idx]

        best = {cat:0 for cat in color_to_category.values()}
        best['overall'] = 0
        for x in range(x1, x2+1):
            col = arr[chart_top + corner_padding:chart_bottom, x]
            cats = [classify_pixel(*px) for px in col]
            bar_rows = [y for y, c in enumerate(cats) if c is not None]
            if len(bar_rows) < 5:
                continue
            total_px = ymax - min(bar_rows)
            if total_px > best['overall']:
                best['overall'] = total_px
                for color, cat_name in color_to_category.items():
                    best[cat_name] = sum(1 for c in cats if c == color)

        # --- After calculating best for each hour ---
        cleaned_best = {
            "overall": best["overall"],                # includes everything
            "social": best.get("social", 0),
            "entertainment": best.get("entertainment", 0)
        }
        result[hour] = cleaned_best


    return result

def process_ios_overall_screenshot(image_path: str):
    # OCR passes
    text = ocr_image(preprocess_for_ocr(image_path))
    light_text = ocr_image(preprocess_for_light_text(image_path))
    lines = [l.strip() for l in (text + "\n" + light_text).split("\n") if l.strip()]

    result = {
        "date": None,
        "is_yesterday": False,
        "total_time": "0h 0m",
        "categories": [],
        "top_apps": [],
        "hourly_usage": {}
    }

    # Date
    for line in lines:
        if "yesterday" in line.lower():
            result["is_yesterday"] = True
            parts = line.split(",")
            result["date"] = parts[-1].strip() if len(parts) > 1 else line.strip()
            break

    # Total time
    in_total_section = False
    for line in lines:
        if "screen time" in line.lower():
            in_total_section = True
            continue
        if not in_total_section: continue
        parsed = parse_time_fragment(line)
        if parsed:
            h, m = parsed
            if h + m > 0:
                result["total_time"] = f"{h}h {m}m"
                break

    # Categories
    for i, line in enumerate(lines):
        if any(cat in line.lower() for cat in ["social", "games", "entertainment"]):
            category_names = line.split()
            if i+1 < len(lines):
                time_matches = re.findall(r"(\d+h\s*\d*m|\d+h|\d*m)", lines[i+1])
                if time_matches and len(time_matches) == len(category_names):
                    for name, tstr in zip(category_names, time_matches):
                        h, m = parse_time_fragment(tstr)
                        result["categories"].append({"name": name, "time": f"{h}h {m}m"})
            break

    # Build color -> category map dynamically
    color_order = ['blue', 'teal', 'orange']
    color_to_category = {}
    for idx, cat in enumerate(result["categories"][:3]):
        color_to_category[color_order[idx]] = cat["name"].lower()
    color_to_category['gray'] = 'other'

    # Top apps
    top_apps = []
    app_lines = {}
    for i, line in enumerate(lines):
        if parse_time_fragment(line): continue
        name = clean_app_name(line.strip())
        if is_valid_app_name(name):
            app_lines[name] = i
    # Find app times
    # Build a map of all times in the OCR lines
    time_map = {}
    for i, line in enumerate(lines):
        parsed = parse_time_fragment(line)
        if parsed:
            h, m = parsed
            if h + m > 0:
                time_map[i] = f"{h}h {m}m"

    for name, idx in app_lines.items():
        for offset in range(1,4):
            if idx+offset in time_map:
                top_apps.append({"name": name, "time": time_map[idx+offset]})
                break
    # Sort descending
    def to_min(tstr): h,m = parse_time_fragment(tstr); return h*60 + m
    result["top_apps"] = sorted(top_apps, key=lambda x: to_min(x["time"]), reverse=True)[:3]

    # Hourly usage
    result["hourly_usage"] = extract_hourly_chart(image_path, color_to_category)

    return result
