from __future__ import annotations

import os
import re
import json
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from unidecode import unidecode

from services.data_loader import load_tour_list, load_tour_detail

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEEDBACK_FILE = os.path.join(os.path.dirname(BASE_DIR), "user_feedback_log.csv")
tour_df: pd.DataFrame = load_tour_list()

CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "bien": ["bien", "bai", "vinh", "dao", "hon", "hai", "cang", "cap treo", "bo bien", "island", "beach", "vịnh", "bãi", "đảo", "hòn", "hải đảo", "cáp treo", "venice", "thi truong bien", "tam bien", "bo cat", "nước biển"],
    "nui": ["nui", "cao nguyen", "doi", "deo", "mang den", "ta dung", "pleiku", "kontum", "kon tum", "banah", "fansipan", "langbiang", "yok don", "sapa", "tay nguyen", "nuoc non", "thac", "rung", "moc chau", "mù cang chải"],
    "thanh_pho": ["thanh pho", "pho", "pho co", "do thi", "city", "khu pho", "urban", "vinwonders", "vinpearl", "grand world", "shopping", "cho", "thi tran", "quan", "quan trung tam", "ben cang", "cau", "san bay", "resort", "khu dan cu", "quang truong", "pho di bo"]
}

try:
    from pydantic import BaseModel

    class RecommendRequest(BaseModel):
        ngan_sach: Optional[int] = None
        so_thich: List[str] = []
        location: Optional[str] = None
        duration: Optional[int] = None
        rating: Optional[float] = None
except Exception:
    BaseModel = object
    RecommendRequest = object

def clean_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return unidecode(text).lower().strip()

def parse_price(price_str: Any) -> int:
    if not isinstance(price_str, str):
        return 0
    cleaned = re.sub(r"[^0-9]", "", price_str)
    try:
        return int(cleaned)
    except ValueError:
        return 0

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def extract_duration_from_name(name: str) -> Optional[int]:
    if not isinstance(name, str):
        return None
    match = re.search(r"(\d+)n", name.lower())
    if match:
        try:
            return int(match.group(1))
        except (ValueError, TypeError):
            return None
    return None

def _get_tour_overview(row: Dict[str, Any]) -> str:
    detail_file = row.get("tour_detail_file")
    if not detail_file:
        return ""
    try:
        detail = load_tour_detail(detail_file)
    except Exception:
        return ""
    overview = detail.get("overview", "")
    if isinstance(overview, str):
        return overview
    return ""

def _compose_content_text(row: Dict[str, Any]) -> str:
    parts = [
        row.get("tour_name", ""),
        row.get("ribbon", ""),
        row.get("region_guess", ""),
        _get_tour_overview(row)
    ]
    combined = " ".join(filter(None, parts))
    return clean_text(combined)

def _guess_categories(content_text: str) -> List[str]:
    categories: List[str] = []
    if not content_text:
        return categories
    text_lower = content_text
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            kw_clean = clean_text(kw)
            if kw_clean and kw_clean in text_lower:
                categories.append(cat)
                break
    return categories

tour_df["content_text"] = tour_df.apply(_compose_content_text, axis=1)
_text_corpus: List[str] = tour_df["content_text"].tolist()
_tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000)
_tfidf_matrix = _tfidf_vectorizer.fit_transform(_text_corpus)
tour_df["categories"] = tour_df["content_text"].apply(_guess_categories)

_region_classifier: Optional[LogisticRegression] = None
try:
    _train_mask = tour_df["region_guess"].notna()
    if _train_mask.sum() >= 10:
        _region_train_X = _tfidf_matrix[_train_mask.values]
        _region_train_y = tour_df.loc[_train_mask, "region_guess"].values
        if len(set(_region_train_y)) > 1:
            rc = LogisticRegression(max_iter=300, multi_class='ovr')
            rc.fit(_region_train_X, _region_train_y)
            _region_classifier = rc
        else:
            _region_classifier = None
except Exception:
    _region_classifier = None

def _predict_region_for_row(row: pd.Series) -> Optional[str]:
    if _region_classifier is None:
        return row.get("region_guess")
    try:
        vec = _tfidf_vectorizer.transform([row.get("content_text", "")])
        pred = _region_classifier.predict(vec)
        return str(pred[0]) if len(pred) else row.get("region_guess")
    except Exception:
        return row.get("region_guess")

tour_df["predicted_region"] = tour_df.apply(_predict_region_for_row, axis=1)

_feature_names = [
    "similarity_score",
    "budget_score",
    "rating_norm",
    "location_score",
    "duration_score",
    "category_score"
]

_feedback_model: Optional[LogisticRegression] = None

def _load_feedback_and_train() -> None:
    global _feedback_model
    if not os.path.exists(FEEDBACK_FILE):
        return
    try:
        df = pd.read_csv(FEEDBACK_FILE)
    except Exception:
        return
    if not set(_feature_names + ["liked"]).issubset(df.columns):
        return
    if len(df) < 20:
        return
    X = df[_feature_names].values
    y = df["liked"].astype(int).values
    model = LogisticRegression(max_iter=200)
    try:
        model.fit(X, y)
        _feedback_model = model
    except Exception:
        _feedback_model = None

_load_feedback_and_train()

DEFAULT_WEIGHTS: Dict[str, float] = {
    "similarity_score": 0.35,
    "budget_score": 0.15,
    "rating_norm": 0.15,
    "location_score": 0.1,
    "duration_score": 0.1,
    "category_score": 0.15
}

def _compute_feature_vector(row: pd.Series, user_profile: Dict[str, Any], similarity: float) -> Dict[str, float]:
    features: Dict[str, float] = {}
    features["similarity_score"] = float(similarity)
    user_budget: Optional[int] = user_profile.get("ngan_sach")
    if user_budget is not None and user_budget > 0:
        price = parse_price(row.get("price", 0))
        if price <= user_budget:
            features["budget_score"] = max(0.0, 1.0 - (user_budget - price) / user_budget)
        else:
            features["budget_score"] = 0.0
    else:
        features["budget_score"] = 0.5
    rating_raw = safe_float(row.get("score", 0), 0.0)
    features["rating_norm"] = min(max(rating_raw / 10.0, 0.0), 1.0)
    desired_loc = user_profile.get("location")
    if desired_loc:
        loc_clean = clean_text(desired_loc)
        region_source = row.get("predicted_region") or row.get("region_guess")
        region_clean = clean_text(region_source or "")
        features["location_score"] = 1.0 if loc_clean in region_clean and region_clean else 0.0
    else:
        features["location_score"] = 0.5
    desired_duration = user_profile.get("duration")
    tour_duration = extract_duration_from_name(row.get("tour_name", ""))
    if desired_duration and tour_duration:
        diff = abs(desired_duration - tour_duration)
        features["duration_score"] = max(0.0, 1.0 - diff / float(desired_duration))
    elif desired_duration and not tour_duration:
        features["duration_score"] = 0.5
    else:
        features["duration_score"] = 0.5
    desired_cats: set[str] = set()
    interests = user_profile.get("so_thich", []) or []
    for interest in interests:
        interest_clean = clean_text(interest)
        if "bien" in interest_clean or "biển" in interest_clean:
            desired_cats.add("bien")
        if "nui" in interest_clean or "núi" in interest_clean:
            desired_cats.add("nui")
        if "thanh pho" in interest_clean or "thành phố" in interest_clean or "city" in interest_clean:
            desired_cats.add("thanh_pho")
        for cat, keywords in CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if clean_text(kw) in interest_clean:
                    desired_cats.add(cat)
                    break
    if desired_cats:
        tour_cats = set(row.get("categories", []) or [])
        if tour_cats:
            common = desired_cats & tour_cats
            features["category_score"] = len(common) / len(desired_cats)
        else:
            features["category_score"] = 0.0
    else:
        features["category_score"] = 0.5
    return features

def normalize_weights(features: Dict[str, float], weights: Dict[str, float]) -> Dict[str, float]:
    valid_keys = [k for k in weights if k in features and features[k] is not None]
    total = sum(weights[k] for k in valid_keys)
    if total == 0 or not valid_keys:
        return {k: 1.0 / len(features) for k in features}
    return {k: weights[k] / total for k in valid_keys}

def _compute_match_score(features: Dict[str, float]) -> float:
    if _feedback_model is not None:
        vec = np.array([features.get(name, 0.0) for name in _feature_names], dtype=float).reshape(1, -1)
        try:
            prob = float(_feedback_model.predict_proba(vec)[0, 1])
            return prob
        except Exception:
            pass
    norm_weights = normalize_weights(features, DEFAULT_WEIGHTS)
    score = sum(norm_weights.get(k, 0.0) * features.get(k, 0.0) for k in norm_weights)
    return float(score)

def recommend_for_user(user_profile: Dict[str, Any], top_n: int = 5) -> List[Dict[str, Any]]:
    interests: List[str] = user_profile.get("so_thich", []) or []
    interest_text = clean_text(" ".join(interests))
    if interest_text:
        user_vec = _tfidf_vectorizer.transform([interest_text])
    else:
        user_vec = np.zeros((1, _tfidf_matrix.shape[1]))
    try:
        similarities = cosine_similarity(user_vec, _tfidf_matrix)[0]
    except Exception:
        similarities = np.zeros(len(tour_df))
    results: List[Dict[str, Any]] = []
    for idx, row in tour_df.iterrows():
        similarity = float(similarities[idx]) if idx < len(similarities) else 0.0
        features = _compute_feature_vector(row, user_profile, similarity)
        user_budget: Optional[int] = user_profile.get("ngan_sach")
        if user_budget is not None and parse_price(row.get("price", 0)) > user_budget:
            continue
        min_rating: Optional[float] = user_profile.get("rating")
        if min_rating is not None and safe_float(row.get("score", 0)) < min_rating:
            continue
        final_score = _compute_match_score(features)
        results.append({
            "tour_id": row.get("tour_id"),
            "tour_name": row.get("tour_name"),
            "match_score": round(final_score, 4),
            "price": row.get("price"),
            "link": row.get("link")
        })
    results_sorted = sorted(results, key=lambda x: x["match_score"], reverse=True)
    return results_sorted[: top_n]
