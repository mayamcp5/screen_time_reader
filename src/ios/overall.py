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

def preprocess_for_ocr(image_path: str) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.grayscale(img)

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.2)

    img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    img_np = np.array(img)

    _, img_np = cv2.threshold(img_np, 180, 255, cv2.THRESH_BINARY_INV)
    return Image.fromarray(img_np)


def preprocess_for_light_text(image_path: str) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.grayscale(img)

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.2)

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.3)

    img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    img_np = np.array(img)

    _, img_np = cv2.threshold(img_np, 150, 255, cv2.THRESH_BINARY_INV)
    return Image.fromarray(img_np)


# ================================
# PIXEL CLASSIFICATION
# ================================

HOURS = [
    '12am','1am','2am','3am','4am','5am','6am','7am','8am','9am','10am','11am',
    '12pm','1pm','2pm','3pm','4pm','5pm','6pm','7pm','8pm','9pm','10pm','11pm'
]


def classify_pixel(r, g, b):
    r, g, b = int(r), int(g), int(b)

    if r < 80 and 100 <= g <= 160 and b > 220:
        return 'social'
    if r > 210 and 140 <= g <= 190 and b < 90:
        return 'entertainment'
    if r < 150 and g > 170 and 180 < b < 240:
        return 'games'
    if 48 <= r <= 68 and 48 <= g <= 68 and 48 <= b <= 68 and abs(r-g) < 5 and abs(g-b) < 5:
        return 'other'
    return None


def is_chart_bg(r, g, b):
    return 20 <= int(r) <= 45 and 20 <= int(g) <= 45 and 20 <= int(b) <= 50


def is_gridline_pixel(r, g, b):
    r, g, b = int(r), int(g), int(b)

    # Must be gray-ish
    if abs(r - g) > 8 or abs(g - b) > 8:
        return False

    # Must be brighter than background but darker than bars
    brightness = (r + g + b) / 3

    # Background ~ 20–45
    # Bars are bright colors
    # Gridlines usually ~ 55–95
    return 50 <= brightness <= 110


# ================================
# HOURLY CHART EXTRACTION (FIXED)
# ================================

def extract_hourly_chart(image_path: str) -> dict:

    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    img_h, img_w = arr.shape[:2]

    # --- Find bar regions vertically ---
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

    best_regions = []
    for probe_x in range(img_w // 3, 2 * img_w // 3, 40):
        regions = find_bar_regions(probe_x)
        if len(regions) > len(best_regions):
            best_regions = regions

    if not best_regions:
        return {}

    chart_start, chart_end = best_regions[-1]
    chart_bottom = chart_end + 1

    # --- Find chart left/right ---
    y_probe = chart_bottom - 5
    chart_left = chart_right = None

    for x in range(img_w):
        if is_chart_bg(*arr[y_probe, x]) and chart_left is None:
            chart_left = x

    for x in range(img_w - 1, -1, -1):
        if is_chart_bg(*arr[y_probe, x]) and chart_right is None:
            chart_right = x

    if chart_left is None or chart_right is None:
        return {}

    chart_mid_x = (chart_left + chart_right) // 2

    # --- Detect horizontal gridlines ---
    gridlines = []
    for y in range(chart_start - 100, chart_bottom):
        if y < 0 or y >= img_h:
            continue

        row = arr[y, chart_left:chart_right]
        gray_count = sum(1 for px in row if is_gridline_pixel(*px))

        if gray_count > 0.4 * (chart_right - chart_left):
            gridlines.append(y)

    # Collapse consecutive rows
    collapsed = []
    if gridlines:
        group = [gridlines[0]]
        for y in gridlines[1:]:
            if y - group[-1] <= 2:
                group.append(y)
            else:
                collapsed.append(int(sum(group) / len(group)))
                group = [y]
        collapsed.append(int(sum(group) / len(group)))

    collapsed = sorted(collapsed)

    print("Detected gridlines:", collapsed)

    if len(collapsed) >= 2:
        chart_top_line = collapsed[0]        # highest gridline
        chart_bottom_line = collapsed[-1]    # baseline
    else:
        chart_top_line = chart_start
        chart_bottom_line = chart_bottom

    ymax = chart_bottom_line - chart_top_line

    # --- Detect bar segments ---
    bar_segments = []
    in_bar = False
    seg_start = None

    for x in range(chart_left, chart_right):
        col = arr[chart_top_line:chart_bottom_line, x]
        has_bar = any(classify_pixel(*px) is not None for px in col)

        if has_bar and not in_bar:
            seg_start = x
            in_bar = True
        elif not has_bar and in_bar:
            if x - seg_start >= 10:
                bar_segments.append((seg_start, x - 1))
            in_bar = False

    if in_bar and chart_right - seg_start >= 10:
        bar_segments.append((seg_start, chart_right - 1))

    result = {'ymax_pixels': ymax}
    for hour in HOURS:
        result[hour] = {
            'overall': 0,
            'social': 0,
            'games': 0,
            'entertainment': 0,
            'other': 0
        }

    if not bar_segments:
        return result

    # --- Compute spacing ---
    if len(bar_segments) >= 2:
        centers = [(x1 + x2) / 2 for x1, x2 in bar_segments]
        spacings = [centers[i+1] - centers[i] for i in range(len(centers)-1)]
        bar_spacing = sum(spacings) / len(spacings)
    else:
        bar_spacing = (chart_right - chart_left) / 24

    first_center = (bar_segments[0][0] + bar_segments[0][1]) / 2
    approx_hour = round((first_center - chart_left) / ((chart_right - chart_left) / 24))
    hour_0_x = first_center - approx_hour * bar_spacing

    for x1, x2 in bar_segments:
        bar_center = (x1 + x2) / 2
        slot_idx = max(0, min(23, round((bar_center - hour_0_x) / bar_spacing)))
        hour = HOURS[slot_idx]

        best = {'overall': 0, 'social': 0, 'games': 0, 'entertainment': 0, 'other': 0}

        for x in range(x1, x2 + 1):
            col = arr[chart_top_line:chart_bottom_line, x]
            cats = [classify_pixel(*px) for px in col]

            bar_rows = [y for y, c in enumerate(cats) if c is not None]
            if len(bar_rows) < 5:
                continue

            bar_top = chart_top_line + min(bar_rows)
            bar_height = chart_bottom_line - bar_top

            if bar_height > best['overall']:
                best['overall'] = bar_height
                best['social'] = sum(1 for c in cats if c == 'social')
                best['games'] = sum(1 for c in cats if c == 'games')
                best['entertainment'] = sum(1 for c in cats if c == 'entertainment')
                best['other'] = sum(1 for c in cats if c == 'other')

        result[hour] = best

    return result


def process_ios_overall_screenshot(image_path: str):
    # First pass: normal preprocessing
    processed_img = preprocess_for_ocr(image_path)
    text = ocr_image(processed_img)

    print("\n================ NORMAL OCR TEXT ================\n")
    print(text)
    print("\n================================================\n")

    # Second pass: aggressive preprocessing for very light gray text
    light_img = preprocess_for_light_text(image_path)
    light_text = ocr_image(light_img)
    
    print("\n================ LIGHT TEXT OCR ================\n")
    print(light_text)
    print("\n===============================================\n")

    # Combine both OCR results
    combined_text = text + "\n" + light_text
    lines = [l.strip() for l in combined_text.split("\n") if l.strip()]

    result = {
        "date": None,
        "is_yesterday": False,
        "total_time": "0h 0m",
        "categories": [],
        "top_apps": [],
        "ymax_pixels": None,
        "hourly_usage": {}
    }

    # -----------------------------
    # 1️⃣ Find date and yesterday
    # -----------------------------
    for line in lines:
        if "yesterday" in line.lower():
            result["is_yesterday"] = True
            parts = line.split(",")
            result["date"] = parts[-1].strip() if len(parts) > 1 else line.strip()
            break

    # -----------------------------
    # 2️⃣ Total screen time
    # -----------------------------
    in_total_section = False
    for line in lines:
        if "screen time" in line.lower():
            in_total_section = True
            continue
        if not in_total_section:
            continue
        parsed = parse_time_fragment(line)
        if parsed:
            h, m = parsed
            if h + m > 0:
                result["total_time"] = f"{h}h {m}m"
                break

    # -----------------------------
    # 3️⃣ Category times
    # -----------------------------
    for i, line in enumerate(lines):
        if any(cat in line.lower() for cat in ["social", "games", "entertainment"]):
            category_names = line.split()
            if i + 1 < len(lines):
                times_line = lines[i + 1]
                time_matches = re.findall(r"(\d+h\s*\d*m|\d+h|\d*m)", times_line)
                if time_matches and len(time_matches) == len(category_names):
                    for name, tstr in zip(category_names, time_matches):
                        h, m = parse_time_fragment(tstr)
                        result["categories"].append({
                            "name": name,
                            "time": f"{h}h {m}m"
                        })
            break

    # -----------------------------
    # 4️⃣ Top apps (first 3 with valid times)
    # -----------------------------
    top_apps = []
    app_candidates = []

    # Find "MOST USED SHOW CATEGORIES" section - look for ALL occurrences
    print("\n🔍 DEBUG: Looking for apps in combined text...\n")
    
    most_used_indices = []
    for i, line in enumerate(lines):
        if "most used" in line.lower():
            most_used_indices.append(i)
            print(f"   Found 'MOST USED' at line {i}: '{line}'")
    
    # Process apps from BOTH "MOST USED" sections (normal + light OCR)
    all_app_entries = {}  # {app_name: [line_indices]}
    
    for most_used_idx in most_used_indices:
        in_top_apps = False
        for i, line in enumerate(lines):
            if i == most_used_idx:
                in_top_apps = True
                continue
            if not in_top_apps or i <= most_used_idx:
                continue
            # Stop at next section headers or UI chrome
            stop_words = ["screen time", "updated", "week", "show categor", "yesterday",
                          "today", "app limits", "always allowed", "content &", "see all"]
            if any(s in line.lower() for s in stop_words) or i > most_used_idx + 10:
                break

            # Skip if this line is itself a time (don't treat times as app names)
            if parse_time_fragment(line):
                continue

            # Clean app name
            cleaned = line.strip()
            # Remove leading OCR icon artifacts: "A.", "eS ", "G ", "&3 ", "A. " etc.
            cleaned = re.sub(r'^[a-zA-Z&\d]{1,2}[\.\s]\s*', '', cleaned)
            cleaned = re.sub(r"['\-,\.]+$", '', cleaned)   # Remove trailing punctuation
            cleaned = re.sub(r"^['\"]|['\"]$", '', cleaned) # Remove leading/trailing quotes
            # Strip trailing digit(s) that are OCR badge counts (e.g. "TikTok 5")
            cleaned = re.sub(r'\s+\d+$', '', cleaned)
            cleaned = cleaned.strip()

            # Skip lines that look like garbled OCR or timestamps
            # e.g. "2n 5/m", "10:11" — slash, colon, or digit-letter combos
            if re.search(r'[\\/:]', cleaned) or re.search(r'\d+[a-zA-Z]', cleaned):
                continue
            
            # Pass through clean_app_name
            name = clean_app_name(cleaned)
            if not is_valid_app_name(name) or len(name) < 3:
                continue

            # Skip obvious non-app-name patterns
            if any(s in name.lower() for s in ["show", "screen", "yesterday", "today",
                                                "february", "january", "march", "april",
                                                "may", "june", "july", "august", "september",
                                                "october", "november", "december"]):
                continue
            
            # Final normalization - strip trailing non-alphanumeric
            normalized_name = name.strip()
            while normalized_name and not normalized_name[-1].isalnum():
                normalized_name = normalized_name[:-1]
            normalized_name = normalized_name.strip()
            if len(normalized_name) < 3:
                continue
            
            # Debug output - show ALL app processing
            if 'instagram' in line.lower() or 'roblox' in line.lower() or 'tiktok' in line.lower():
                print(f"      Line {i}: '{line}' → name: '{name}' (len={len(name)}, repr={repr(name)}) → normalized: '{normalized_name}' (len={len(normalized_name)})")
            
            # Store this occurrence
            if normalized_name not in all_app_entries:
                all_app_entries[normalized_name] = []
            all_app_entries[normalized_name].append(i)
    
    print(f"\n📱 Found apps: {list(all_app_entries.keys())}\n")

    # Build a map of ALL time values in combined text
    time_map = {}
    print("🔍 DEBUG: Scanning for ALL times in combined text...")
    for i, line in enumerate(lines):
        parsed = parse_time_fragment(line)
        if parsed:
            h, m = parsed
            if h + m > 0:
                time_str = f"{h}h {m}m"
                time_map[i] = time_str
                print(f"   Line {i}: '{line}' → {time_str}")
    print(f"\n📊 Total times found: {len(time_map)}\n")

    # Match apps with their times by looking near ANY occurrence of the app
    print("🔗 DEBUG: Matching apps with times...\n")
    for app_name, line_indices in all_app_entries.items():
        if len(top_apps) >= 3:
            break
        
        app_time = "0h 0m"
        print(f"   '{app_name}' found at lines {line_indices}:")
        
        # Check occurrences in REVERSE order (so light OCR / later lines get priority)
        for name_idx in reversed(line_indices):
            if app_time != "0h 0m":
                break  # Already found a time
            
            # Look in next few lines after this occurrence
            for offset in range(1, 4):
                check_idx = name_idx + offset
                if check_idx in time_map:
                    app_time = time_map[check_idx]
                    print(f"      FOUND {app_time} at line {check_idx} (checking from line {name_idx})")
                    # Remove this time so it's not reused
                    del time_map[check_idx]
                    break
        
        if app_time == "0h 0m":
            print(f"      ⚠️ No time found for {app_name}")
        
        top_apps.append({"name": app_name, "time": app_time})

    # Sort top apps by usage time descending
    def time_to_minutes(tstr):
        h, m = parse_time_fragment(tstr)
        return h * 60 + m

    top_apps.sort(key=lambda x: time_to_minutes(x["time"]), reverse=True)
    result["top_apps"] = top_apps[:3]

    # -----------------------------
    # 5️⃣ Hourly usage chart
    # -----------------------------
    hourly_raw = extract_hourly_chart(image_path)
    ymax = hourly_raw.pop('ymax_pixels', None)
    result["ymax_pixels"] = ymax
    # Only expose the three RA-tracked categories per hour
    result["hourly_usage"] = {
        hour: {
            'overall':       data['overall'],
            'social':        data['social'],
            'entertainment': data['entertainment'],
        }
        for hour, data in hourly_raw.items()
    }

    return result