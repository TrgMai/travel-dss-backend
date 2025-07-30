# data_loader.py
import json
import os
import pandas as pd
import re

# === Config ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOUR_JSON = os.path.join(BASE_DIR, "data", "all-tours.json")
TOUR_DETAIL_FOLDER = os.path.join(BASE_DIR, "data", "tour-detail")
HOTEL_FOLDER = os.path.join(BASE_DIR, "data", "hotels")


# === Region Mapping ===
REGION_KEYWORDS = {
    "da-lat": ["đà lạt", "lâm đồng"],
    "da-nang": ["đà nẵng"],
    "nha-trang": ["nha trang", "khánh hòa"],
    "phan-thiet": ["phan thiết", "mũi né", "bình thuận"],
    "phu-quoc": ["phú quốc", "kiên giang"],
    "quy-nhon": ["quy nhơn", "bình định"],
    "tinh-phu-yen": ["phú yên", "tuy hòa"],
    "vung-tau": ["vũng tàu", "bà rịa"],
    "can-tho": ["cần thơ", "miền tây", "bạc liêu", "cà mau", "hậu giang", "long xuyên", "sóc trăng"],
    "tay-nguyen": ["tây nguyên", "buôn mê thuột", "buôn ma thuột", "gia lai", "kontum", "kon tum", "măng đen", "tà đùng", "đắk lắk", "đăk lăk", "pleiku"]
}

# === Functions ===
def extract_tour_id(link):
    match = re.search(r"/(\d+)$", link)
    return match.group(1) if match else None

def guess_region(tour_name):
    name_lower = tour_name.lower()
    for region, keywords in REGION_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return region
    return None

def load_tour_list(json_path=TOUR_JSON):
    with open(json_path, "r", encoding="utf-8") as f:
        all_tours = json.load(f)

    tour_entries = []
    for tour in all_tours:
        tour_id = extract_tour_id(tour.get("link", ""))
        name = tour.get("name", "")
        region = guess_region(name)
        tour_entries.append({
            "tour_id": tour_id,
            "tour_name": name,
            "link": tour.get("link"),
            "price": tour.get("price"),
            "departure_date": tour.get("departure_date"),
            "score": tour.get("score"),
            "region_guess": region,
            "tour_detail_file": f"tour-{tour_id}.json" if tour_id else None,
            "hotel_file_guess": f"khach-san-{region}.json" if region else None
        })

    return pd.DataFrame(tour_entries)

def load_tour_detail(tour_detail_file):
    path = os.path.join(TOUR_DETAIL_FOLDER, tour_detail_file)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_hotel_list(hotel_file):
    path = os.path.join(HOTEL_FOLDER, hotel_file)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# === Example usage ===
if __name__ == "__main__":
    df = load_tour_list()
    print(df.head())
