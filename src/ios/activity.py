# src/ios/activity.py

from src.utils import ocr_image
from src.parsing.time_parsing import parse_time_fragment
from src.parsing.app_name_parsing import clean_app_name, is_valid_app_name


def process_ios_category_screenshot(image_path: str):

    text = ocr_image(image_path)

    print("\n================ RAW OCR TEXT ================\n")
    print(text)
    print("\n=============================================\n")

    lines = [l.strip() for l in text.split("\n") if l.strip()]

    category = None
    total_time = "0h 0m"
    apps = []

    # ======================================================
    # 1️⃣ Detect category
    # ======================================================
    for line in lines:
        lower = line.lower()
        if "entertainment" in lower:
            category = "Entertainment"
        if "social" in lower:
            category = "Social"

    # ======================================================
    # 2️⃣ Extract TOTAL TIME
    # Only between "SCREEN TIME" and "APPS & WEBSITES"
    # ======================================================
    in_total_section = False

    for line in lines:
        lower = line.lower()

        if "screen time" in lower:
            in_total_section = True
            continue

        if "apps & websites" in lower:
            break

        if not in_total_section:
            continue

        if "daily" in lower:
            continue

        line = line.replace("Th", "1h").replace("lh", "1h").replace("Ih", "1h")
        parsed = parse_time_fragment(line)
        if parsed:
            h, m = parsed
            if h > 0 or m > 0:
                total_time = f"{h}h {m}m"
                break

    # ======================================================
    # 3️⃣ Slice APPS section and TIMES section cleanly
    # ======================================================

    apps_section = []
    times_section = []

    in_apps = False
    in_times = False

    for line in lines:

        lower = line.lower()

        if "apps & websites" in lower:
            in_apps = True
            continue

        if "limits" in lower:
            in_apps = False
            in_times = True
            continue

        if in_apps:
            apps_section.append(line)

        elif in_times:
            times_section.append(line)

    # ======================================================
    # 4️⃣ Detect layout type
    # ======================================================

    # If apps_section lines contain times → same-line layout
    same_line_layout = any(parse_time_fragment(l) for l in apps_section)

    # ======================================================
    # 5️⃣ Parse SAME-LINE layout (Entertainment style)
    # ======================================================
    if same_line_layout:

        for line in apps_section:

            line = line.replace("Th", "1h").replace("lh", "1h").replace("Ih", "1h")
            parsed = parse_time_fragment(line)
            if not parsed:
                continue

            h, m = parsed

            if h == 0 and m == 0:
                continue

            # Remove time from name
            name_part = line
            name_part = name_part.replace(f"{h}h", "")
            name_part = name_part.replace(f"{m}m", "")
            name_part = clean_app_name(name_part)

            if not is_valid_app_name(name_part):
                continue

            apps.append({
                "name": name_part,
                "time": f"{h}h {m}m"
            })

    # ======================================================
    # 6️⃣ Parse SPLIT layout (Social style)
    # ======================================================
    else:

        names = []
        times = []

        # Collect names
        for line in apps_section:
            line = line.replace("Th", "1h").replace("lh", "1h").replace("Ih", "1h")
            parsed = parse_time_fragment(line)
            if parsed:
                continue

            name = clean_app_name(line)

            if not is_valid_app_name(name):
                continue

            if len(name) < 3:
                continue

            names.append(name)

        # Collect times
        for line in times_section:

            parsed = parse_time_fragment(line)
            if not parsed:
                continue

            h, m = parsed

            if h == 0 and m == 0:
                continue

            times.append((h, m))

        # Remove first time (Daily Average)
        if times:
            times = times[1:]


        for name, (h, m) in zip(names, times):
            apps.append({
                "name": name,
                "time": f"{h}h {m}m"
            })

    # ======================================================
    # 7️⃣ Sort apps descending by minutes
    # ======================================================
    apps.sort(
        key=lambda x: int(x["time"].split("h")[0]) * 60 +
                      int(x["time"].split("h")[1].replace("m", "").strip()),
        reverse=True
    )

    return {
        "category": category,
        "total_time": total_time,
        "apps": apps
    }
