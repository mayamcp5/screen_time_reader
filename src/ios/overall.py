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

    # Convert to grayscale
    img = ImageOps.grayscale(img)

    # Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)

    # Increase brightness slightly
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.2)

    # Optional: resize to make small text more legible
    img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)

    # Convert to numpy array for OpenCV processing
    img_np = np.array(img)

    # Threshold: make light gray text dark
    _, img_np = cv2.threshold(img_np, 180, 255, cv2.THRESH_BINARY_INV)

    return Image.fromarray(img_np)

def preprocess_for_light_text(image_path: str) -> Image.Image:
    """Specifically optimized for very light gray text like the times."""
    img = Image.open(image_path).convert("RGB")
    
    # Convert to grayscale
    img = ImageOps.grayscale(img)
    
    # Moderate contrast boost for faint text
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.2)
    
    # Slightly more brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.3)
    
    # Standard resize
    img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    
    img_np = np.array(img)
    
    # Lower threshold to catch lighter gray text (150 instead of 180)
    _, img_np = cv2.threshold(img_np, 150, 255, cv2.THRESH_BINARY_INV)
    
    return Image.fromarray(img_np)

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
        "categories": [],   # [{"name": "Social", "time": "5h 38m"}]
        "top_apps": []      # [{"name": "Instagram", "time": "3h 45m"}]
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
            # Stop at unrelated UI junk or next section
            if any(skip in line.lower() for skip in ["screen time", "updated", "week"]) or i > most_used_idx + 15:
                break

            # Skip if this line is itself a time (don't treat times as app names)
            if parse_time_fragment(line):
                continue

            # Clean app name
            cleaned = line.strip()
            cleaned = re.sub(r'^[a-zA-Z]{1,2}\s+', '', cleaned)  # Remove leading OCR artifacts
            cleaned = re.sub(r"['\-,\.]+$", '', cleaned)  # Remove trailing punctuation
            cleaned = re.sub(r"^['\"]|['\"]$", '', cleaned)  # Remove leading/trailing quotes
            cleaned = cleaned.strip()
            
            # Pass through clean_app_name
            name = clean_app_name(cleaned)
            if not is_valid_app_name(name) or len(name) < 3:
                continue
            
            # Final normalization - aggressively remove trailing junk
            # First strip whitespace
            normalized_name = name.strip()
            # Then remove any trailing non-alphanumeric characters (except internal spaces)
            while normalized_name and not normalized_name[-1].isalnum():
                normalized_name = normalized_name[:-1]
            normalized_name = normalized_name.strip()
            
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

    return result