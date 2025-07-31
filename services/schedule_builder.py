from __future__ import annotations

import json
import os
from typing import List, Optional, Dict, Any

from unidecode import unidecode

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_TOURS_ID_PATH = os.path.join(BASE_DIR, "data", "all_tours_id.json")
HOTEL_FOLDER = os.path.join(BASE_DIR, "data", "hotels")

try:
    from pydantic import BaseModel

    class ScheduleRequest(BaseModel):
        tour_id: str
        location: str
        ngan_sach: Optional[int] = None
        so_thich: List[str] = []
        duration: Optional[int] = None
        rating: Optional[float] = None
except Exception:
    BaseModel = object  # type: ignore
    ScheduleRequest = object  # type: ignore

def _clean_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return unidecode(text).lower().strip()

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def _load_tour_detail(tour_id: str) -> Dict[str, Any]:
    if not os.path.exists(ALL_TOURS_ID_PATH):
        return {}
    try:
        with open(ALL_TOURS_ID_PATH, "r", encoding="utf-8") as f:
            all_tours = json.load(f)
    except Exception:
        return {}
    for tour in all_tours:
        if str(tour.get("id")) == str(tour_id):
            return tour
    return {}

def _extract_schedule(tour_detail: Dict[str, Any]) -> List[Dict[str, Any]]:
    return tour_detail.get("accordion_schedule", []) or []

def _extract_hotels_by_region(region: str) -> List[Dict[str, Any]]:
    if not region:
        return []
    path = os.path.join(HOTEL_FOLDER, f"khach-san-{region}.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def _jaccard_similarity(a: str, b: str) -> float:
    set_a = set(_clean_text(a).split()) if a else set()
    set_b = set(_clean_text(b).split()) if b else set()
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    if not union:
        return 0.0
    return len(intersection) / len(union)

def _score_hotel(hotel: Dict[str, Any], user_keywords: List[str], budget: Optional[int]) -> float:
    rating_norm = min(max(_safe_float(hotel.get("rating")) / 10.0, 0.0), 1.0)
    if budget is not None and budget > 0:
        price_val = _safe_int(hotel.get("price"))
        budget_score = 1.0 if price_val <= budget else 0.0
    else:
        budget_score = 0.5
    user_text = " ".join(user_keywords) if user_keywords else ""
    name_similarity = _jaccard_similarity(user_text, hotel.get("name", ""))
    return 0.5 * rating_norm + 0.3 * budget_score + 0.2 * name_similarity

def build_schedule(req: ScheduleRequest) -> Dict[str, Any]:
    tour_detail = _load_tour_detail(req.tour_id)
    if not tour_detail:
        return {"error": "Tour not found"}
    raw_schedule = _extract_schedule(tour_detail)
    user_keywords: List[str] = [
        _clean_text(word) for word in (req.so_thich or []) if _clean_text(word)
    ]
    enriched_schedule: List[Dict[str, Any]] = []
    for entry in raw_schedule:
        day_label = entry.get("day", "")
        detail_text = entry.get("detail", "")
        combined_text = f"{day_label} {detail_text}"
        interest_score = 0.0
        if user_keywords:
            interest_score = _jaccard_similarity(" ".join(user_keywords), combined_text)
        enriched_schedule.append({
            "day": day_label,
            "detail": detail_text,
            "interest_score": round(interest_score, 4)
        })
    hotels = _extract_hotels_by_region(req.location)
    if req.rating is not None:
        hotels = [h for h in hotels if _safe_float(h.get("rating")) >= req.rating]
    if req.ngan_sach is not None:
        hotels = [h for h in hotels if _safe_int(h.get("price")) <= req.ngan_sach]
    ranked_hotels = []
    for hotel in hotels:
        score = _score_hotel(hotel, user_keywords, req.ngan_sach)
        ranked_hotels.append({
            "name": hotel.get("name"),
            "location": hotel.get("location"),
            "price": hotel.get("price"),
            "rating": hotel.get("rating"),
            "link": hotel.get("link"),
            "images": hotel.get("images"),
            "video_thumbnail": hotel.get("video_thumbnail"),
            "score": round(score, 4)
        })
    ranked_hotels.sort(key=lambda x: x["score"], reverse=True)
    response: Dict[str, Any] = {
        "tour": {
            "id": str(req.tour_id),
            "name": tour_detail.get("name"),
            "url": tour_detail.get("url"),
            "price": tour_detail.get("price_adult"),
            "gallery": tour_detail.get("gallery"),
            "overview": tour_detail.get("overview")
        },
        "schedule": enriched_schedule,
        "hotels": ranked_hotels[:5]
    }
    return response
