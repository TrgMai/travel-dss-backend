# recommend_tours.py
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data_loader import load_tour_list
import re
from unidecode import unidecode

# === Load dữ liệu tour ===
tour_df = load_tour_list()

# === Hàm làm sạch văn bản ===
def clean_text(text):
    if not isinstance(text, str): return ""
    return unidecode(text).lower().strip()

# === Vector hoá content tour ===
def preprocess_tour_text(row):
    region = row.get("region_guess") or ""
    region = region.replace("-", " ") if isinstance(region, str) else ""
    parts = [
        row.get("tour_name", ""),
        row.get("ribbon", ""),
        region
    ]
    return clean_text(" ".join(parts))

tour_df["text_vector"] = tour_df.apply(preprocess_tour_text, axis=1)

vectorizer = TfidfVectorizer()
tour_tfidf_matrix = vectorizer.fit_transform(tour_df["text_vector"])

# === Hàm gợi ý tour ===
def recommend_for_user(user_profile, top_n=5):
    filtered_df = tour_df.copy()

    # === Tính điểm similarity từ sở thích ===
    user_text = clean_text(" ".join(user_profile.get("so_thich", [])))
    user_vector = vectorizer.transform([user_text])
    tour_vectors = vectorizer.transform(filtered_df["text_vector"])
    similarity_scores = cosine_similarity(user_vector, tour_vectors)[0]

    # === Tính điểm cho từng tour ===
    results = []
    for idx, row in filtered_df.iterrows():
        match_score = 0.0

        # 1. Giá gần ngân sách → điểm
        def parse_price(price_str):
            if not price_str: return 0
            return int(price_str.replace(".", "").replace("đ", "").strip())

        if user_profile.get("ngan_sach") is not None:
            price = parse_price(row.get("price", "0"))
            max_budget = user_profile["ngan_sach"]
            if price > max_budget:
                continue  # loại bỏ nếu vượt ngân sách
            else:
                budget_score = 1 - abs(price - max_budget) / max_budget
                budget_score = max(min(budget_score, 1.0), 0.0)
                match_score += budget_score * 0.2

        # 2. Vị trí khớp
        if user_profile.get("location"):
            loc = clean_text(user_profile["location"])
            region = clean_text(row.get("region_guess", ""))
            if loc in region:
                match_score += 0.2

        # 3. Duration khớp
        if user_profile.get("duration"):
            match = re.search(r"(\d+)N", row.get("tour_name", "").upper())
            days = int(match.group(1)) if match else None
            if days == user_profile["duration"]:
                match_score += 0.1

        # 4. Đánh giá đủ cao
        if user_profile.get("rating"):
            try:
                rating = float(row.get("score", 0))
                if rating >= user_profile["rating"]:
                    match_score += 0.1
            except:
                pass

        # 5. Similarity từ sở thích
        similarity = similarity_scores[idx]
        match_score += similarity * 0.4

        results.append({
            "tour_name": row["tour_name"],
            "match_score": round(match_score, 3),
            "price": row["price"],
            "link": row["link"]
        })

    # Sắp xếp theo điểm
    results = sorted(results, key=lambda x: x["match_score"], reverse=True)[:top_n]
    return results

# === FastAPI ===
app = FastAPI()

class RecommendRequest(BaseModel):
    ngan_sach: int = None
    so_thich: list[str] = []
    location: str | None = None
    duration: int | None = None
    rating: float | None = None
