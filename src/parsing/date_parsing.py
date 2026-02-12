import re

SPANISH_TO_ENGLISH_MONTH = {
    "enero": "January",
    "febrero": "February",
    "marzo": "March",
    "abril": "April",
    "mayo": "May",
    "junio": "June",
    "julio": "July",
    "agosto": "August",
    "septiembre": "September",
    "octubre": "October",
    "noviembre": "November",
    "diciembre": "December"
}

ENGLISH_MONTHS = set(SPANISH_TO_ENGLISH_MONTH.values())


def extract_android_date(text: str) -> str:
    # English: March 24
    match_en = re.search(r"\b([A-Z][a-z]+)\s+(\d{1,2})\b", text)
    if match_en and match_en.group(1) in ENGLISH_MONTHS:
        return f"{match_en.group(1)} {match_en.group(2)}"

    # Spanish: 24 de marzo
    match_es = re.search(
        r"\b(\d{1,2})\s+de\s+([a-záéíóúüñ]+)\b",
        text,
        re.IGNORECASE
    )
    if match_es:
        day, month_es = match_es.group(1), match_es.group(2).lower()
        month_en = SPANISH_TO_ENGLISH_MONTH.get(month_es)
        if month_en:
            return f"{month_en} {day}"

    return ""
