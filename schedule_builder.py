from typing import List, Optional
from pydantic import BaseModel
import json
import os


class ScheduleRequest(BaseModel):
    tour_id: str
    location: str  # ví dụ: "phu-quoc"
    ngan_sach: Optional[int] = None
    so_thich: List[str] = []
    duration: Optional[int] = None
    rating: Optional[float] = None

def load_tour_detail(tour_id: str):
    with open("data/all_tours_id.json", "r", encoding="utf-8") as f:
        all_tours = json.load(f)
    for tour in all_tours:
        if tour["id"] == tour_id:
            return tour
    return {}

def extract_schedule(tour_detail: dict):
    return tour_detail.get("accordion_schedule", [])

def extract_hotel_by_region(region: str):
    path = f"data/hotels/khach-san-{region}.json"
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_float(value, default=0.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def build_schedule(req: ScheduleRequest) -> dict:
    tour_detail = load_tour_detail(req.tour_id)
    if not tour_detail:
        return {"error": "Tour not found"}

    schedule = extract_schedule(tour_detail)
    hotels = extract_hotel_by_region(req.location)

    # Lọc theo rating
    if req.rating is not None:
        hotels = [h for h in hotels if safe_float(h.get("rating")) >= req.rating]

    # Lọc theo ngân sách
    if req.ngan_sach is not None:
        hotels = [h for h in hotels if safe_int(h.get("price")) <= req.ngan_sach]

    # Ưu tiên theo rating
    hotels = sorted(hotels, key=lambda h: safe_float(h.get("rating")), reverse=True)

    return {
        "tour": {
            "id": req.tour_id,
            "name": tour_detail.get("name"),
            "url": tour_detail.get("url"),
            "price": tour_detail.get("price_adult"),
            "gallery": tour_detail.get("gallery"),
            "overview": tour_detail.get("overview")
        },
        "schedule": schedule,
        "hotels": hotels[:5]  # lấy top 5 khách sạn phù hợp
    }
