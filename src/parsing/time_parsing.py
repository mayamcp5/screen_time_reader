import re

# Patterns
HOURS_MIN_PATTERN = r"(\d+)\s*h\s*(\d+)\s*(?:m|min)"
HOURS_ONLY_PATTERN = r"\b(\d+)\s*h\b"
MINUTES_ONLY_PATTERN = r"\b(\d+)\s*(?:m|min)\b"

def parse_time_fragment(text: str):
    """
    Parses time strings like:
    - 2h 38 min
    - 2 h
    - 38 min
    Returns (hours, minutes) or None
    """
    # Pre-correct common OCR mistakes in time strings
    # Fix common OCR errors: I, l, t instead of 1
    corrected_text = text
    corrected_text = re.sub(r'\b([Ilt]h)\b', '1h', corrected_text, flags=re.IGNORECASE)
    corrected_text = re.sub(r'\b([Ilt])h\s', '1h ', corrected_text, flags=re.IGNORECASE)
    # Fix Z→2 when followed by digits then m (e.g. "Z6m" → "26m")
    corrected_text = re.sub(r'\bZ(\d+[mM])', r'2\1', corrected_text)
    # Fix uppercase M at end of minute token (e.g. "26mM" → "26m", "Z6mM" → "26m")
    corrected_text = re.sub(r'(\d+[mM])M\b', lambda m: m.group(1).lower(), corrected_text)
    
    # Last occurrence of hours+minutes
    matches = list(re.finditer(HOURS_MIN_PATTERN, corrected_text, re.IGNORECASE))
    if matches:
        last_match = matches[-1]
        return int(last_match.group(1)), int(last_match.group(2))

    # Last occurrence of hours only
    matches = list(re.finditer(HOURS_ONLY_PATTERN, corrected_text, re.IGNORECASE))
    if matches:
        last_match = matches[-1]
        return int(last_match.group(1)), 0

    # Last occurrence of minutes only
    matches = list(re.finditer(MINUTES_ONLY_PATTERN, corrected_text, re.IGNORECASE))
    if matches:
        last_match = matches[-1]
        minutes = int(last_match.group(1))
        # Special fix: if minutes >= 100, split last 2 digits as minutes
        if minutes >= 100:
            h = int(str(minutes)[:-2])
            m = int(str(minutes)[-2:])
            return h, m
        return 0, minutes

    return None