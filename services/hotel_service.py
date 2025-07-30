import json
import os
import unicodedata
from typing import List, Tuple, Optional
from pydantic import BaseModel

# === Định nghĩa đường dẫn gốc dự án ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOTEL_FOLDER = os.path.join(BASE_DIR, "data", "hotels")
ALL_HOTELS_JSON = os.path.join(BASE_DIR, "data", "all_hotels.json")


# === Request Models ===
class HotelSearchRequest(BaseModel):
    location: str

class HotelSearchByNameRequest(BaseModel):
    name: str


# === Utility ===
def normalize_text(text: str) -> str:
    """Chuyển thành chuỗi không dấu, viết thường, bỏ khoảng trắng để so sánh gần đúng"""
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    return text.lower().replace(" ", "").replace("-", "")


# === Search theo vùng ===
def find_best_matching_region(user_input: str) -> Optional[str]:
    normalized_input = normalize_text(user_input)

    if not os.path.exists(HOTEL_FOLDER):
        return None

    for filename in os.listdir(HOTEL_FOLDER):
        if filename.startswith("khach-san-") and filename.endswith(".json"):
            region_slug = filename[len("khach-san-"):-len(".json")]
            if normalized_input in normalize_text(region_slug):
                return region_slug
    return None

def extract_hotels_by_region_slug(region_slug: str) -> List[dict]:
    path = os.path.join(HOTEL_FOLDER, f"khach-san-{region_slug}.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_hotels_by_fuzzy_location(location_input: str) -> Tuple[List[dict], Optional[str]]:
    matched_slug = find_best_matching_region(location_input)
    if not matched_slug:
        return [], None
    hotels = extract_hotels_by_region_slug(matched_slug)
    return hotels, matched_slug


# === Search theo tên khách sạn ===
def load_all_hotels() -> List[dict]:
    if not os.path.exists(ALL_HOTELS_JSON):
        return []
    with open(ALL_HOTELS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def search_hotels_by_name(name: str) -> List[dict]:
    hotels = load_all_hotels()
    target = normalize_text(name)
    matched = [
        hotel for hotel in hotels
        if target in normalize_text(hotel.get("name", ""))
    ]
    return matched
