import re


def clean_app_name(text: str) -> str:
    """
    Removes leading OCR symbols and extra whitespace.
    """
    text = re.sub(r"^[^A-Za-z0-9]+", "", text)
    return text.strip()


def is_valid_app_name(name: str) -> bool:
    """
    Simple validity check to avoid garbage from OCR.
    """
    if len(name) < 2:
        return False
    if name.isdigit():
        return False
    # if re.search(r'[\\/:]', candidate) or re.search(r'\d+[a-zA-Z]', candidate):
    #     return False
    return True
