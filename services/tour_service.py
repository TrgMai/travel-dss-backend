import os
import json
from typing import List

# === Đường dẫn gốc của project ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOUR_JSON = os.path.join(BASE_DIR, "data", "all-tours.json")

def safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def load_all_tours(filepath=TOUR_JSON) -> List[dict]:
    if not os.path.exists(filepath):
        print(f"[!] Không tìm thấy file: {filepath}")
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def get_top_hot_tours(top_n: int = 10) -> List[dict]:
    all_tours = load_all_tours()

    valid_tours = [
        tour for tour in all_tours
        if safe_float(tour.get("score")) > 0
    ]

    sorted_tours = sorted(valid_tours, key=lambda t: safe_float(t.get("score")), reverse=True)
    return sorted_tours[:top_n]


