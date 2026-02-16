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
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    img_h, img_w = arr.shape[:2]

    # 1. Find the Chart Floor & Horizontal Bounds
    # We use a probe to find the vertical region of the bars first
    def find_bar_regions(probe_x):
        regions = []; in_region = False; start = None
        for y in range(img_h // 4, img_h):
            if classify_pixel(*arr[y, probe_x]) is not None:
                if not in_region:
                    start = y; in_region = True
            elif in_region:
                if y - start > 50: regions.append((start, y))
                in_region = False
        return regions

    best_regions = find_bar_regions(img_w // 2)
    if not best_regions: return {}
    
    chart_start, chart_end = best_regions[-1]
    chart_bottom = chart_end + 1
    
    # Use the background color to find the absolute left/right of the grid
    y_probe = chart_bottom - 5
    chart_bg_x = [x for x in range(img_w) if is_chart_bg(*arr[y_probe, x])]
    if not chart_bg_x: return {}
    chart_left, chart_right = min(chart_bg_x), max(chart_bg_x)

    # 2. Fix the Ceiling (Capture the full height)
    # Instead of scanning for 'clean' space, we look for the highest pixel 
    # of ANY bar in the entire chart and add a small buffer.
    absolute_top = chart_bottom
    for x in range(chart_left, chart_right, 5):
        for y in range(chart_bottom, max(0, chart_bottom - 400), -1):
            if classify_pixel(*arr[y, x]):
                if y < absolute_top: absolute_top = y
    
    # Set chart_top slightly above the highest detected bar pixel
    chart_top = max(0, absolute_top - 5)

    # 3. Precise Slot Logic (The 24-Hour Grid)
    # The total width of the grid represents exactly 24 hours.
    slot_width = (chart_right - chart_left) / 24
    result = {hour: {"overall": 0, "social": 0, "entertainment": 0} for hour in HOURS}
    tallest_bar = 0

    for i in range(24):
        hour = HOURS[i]
        # Calculate the center of this hour's slot
        slot_center_x = int(chart_left + (i * slot_width) + (slot_width / 2))
        
        # Scan a few pixels around the center of the slot to find the bar
        max_h = 0
        best_cats = []
        
        # We check 3 columns in the center of the slot to be robust
        for x in range(slot_center_x - 1, slot_center_x + 2):
            if not (0 <= x < img_w): continue
            
            # Look at the column from the bottom up
            col_pixels = []
            for y in range(chart_bottom, chart_top, -1):
                cat = classify_pixel(*arr[y, x])
                if cat:
                    col_pixels.append(cat)
            
            if len(col_pixels) > max_h:
                max_h = len(col_pixels)
                best_cats = col_pixels

        if max_h > 0:
            # Note: iOS bars are always anchored to the bottom. 
            # Total height = distance from bottom to highest detected pixel.
            social_px = round(best_cats.count("blue"))
            ent_px = round(best_cats.count("orange"))
            
            result[hour] = {
                "overall": max_h,
                "social": social_px,
                "entertainment": ent_px
            }
            if max_h > tallest_bar:
                tallest_bar = max_h

    result['ymax_pixels'] = tallest_bar
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
