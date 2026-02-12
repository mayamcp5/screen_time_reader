import re
from src.utils import ocr_image
from src.parsing.time_parsing import parse_time_fragment
from src.parsing.app_name_parsing import clean_app_name, is_valid_app_name

def process_android_activity_history(image_paths):
    """
    Processes one or more Android activity history screenshots.
    Returns:
        {
            "apps": [
                {"name": "TikTok", "time": "2h 38m"},
                {"name": "WhatsApp", "time": "1h 57m"},
                ...
            ]
        }
    """
    app_dict = {}  # Store app -> time, skip duplicates

    for image_path in image_paths:
        text = ocr_image(image_path)
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        i = 0
        while i < len(lines):
            line = lines[i]

            # Only consider lines that are not a time
            if parse_time_fragment(line) is None:
                name = clean_app_name(line)
                if is_valid_app_name(name) and name not in app_dict:
                    # Look ahead for next line(s) containing time
                    for j in range(1, 3):  # check next 1-2 lines
                        if i + j < len(lines):
                            h_m = parse_time_fragment(lines[i + j])
                            if h_m:
                                h, m = h_m
                                app_dict[name] = f"{h}h {m}m"
                                break
            i += 1

    # Convert dict to list and sort by total minutes descending
    apps = []
    for name, time_text in app_dict.items():
        h, m = parse_time_fragment(time_text)
        apps.append({"name": name, "time": f"{h}h {m}m"})

    apps.sort(
        key=lambda x: int(x["time"].split("h")[0]) * 60 + int(x["time"].split("h")[1].replace("m", "").strip()),
        reverse=True
    )

    return {"apps": apps}
