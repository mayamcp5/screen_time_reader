from utils import extract_overall_info_and_bars

def process_overall_screenshot(image_path):
    """
    Handles a single iOS overall screenshot.
    Returns structured data:
      {
        "date": "Feb 5",
        "total_time": "5h 42m",
        "hourly_breakdown": [0, 10, 30, ...]  # minutes per hour
      }
    """
    return extract_overall_info_and_bars(image_path)
