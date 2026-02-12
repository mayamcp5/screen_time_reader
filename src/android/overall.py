import re
from src.utils import ocr_image
from src.parsing.time_parsing import parse_time_fragment
from src.parsing.date_parsing import extract_android_date
from src.parsing.app_name_parsing import clean_app_name, is_valid_app_name

def process_android_overall_screenshot(image_path: str):
    """
    Processes a single Android overall screenshot.
    Returns a dict:
      {
        "date": "March 24",
        "total_time": "7h 52m",
        "top_apps": [{"name": "TikTok", "time": "2h 38m"}, ...]
      }
    """
    text = ocr_image(image_path)
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # 1️⃣ Extract date
    date = extract_android_date(text)

    # 2️⃣ Extract total screen time (take first non-zero time)
    total_time = "0h 0m"
    for line in lines:
        parsed = parse_time_fragment(line)
        if parsed:
            h, m = parsed
            if h + m > 0:
                total_time = f"{h}h {m}m"
                break

    # 3️⃣ Extract top apps
    top_apps = []
    for line in lines:
        # Find all time fragments in the line
        matches = list(re.finditer(
            r"(\d+\s*h\s*\d+\s*(?:m|min))|(\d+\s*h\b)|(\d+\s*(?:m|min)\b)",
            line,
            re.IGNORECASE
        ))
        if not matches:
            continue

        # Take the last match as app time
        last_match = matches[-1]
        time_text = last_match.group(0)
        parsed = parse_time_fragment(time_text)
        if not parsed:
            continue
        h, m = parsed

        # Everything before the last time match is app name
        name_part = line[:last_match.start()].strip()
        # Strip leading non-alphanumeric chars (like @)_ or ©)
        name_part = re.sub(r"^[^\w]+", "", name_part)

        name = clean_app_name(name_part)
        if not is_valid_app_name(name):
            continue

        top_apps.append({
            "name": name,
            "time": f"{h}h {m}m"
        })

    # Sort apps by total minutes descending
    top_apps.sort(
        key=lambda x: int(x["time"].split("h")[0]) * 60 + int(x["time"].split("h")[1].replace("m", "").strip()),
        reverse=True
    )

    return {
        "date": date,
        "total_time": total_time,
        "top_apps": top_apps
    }
