# Gói cần thiết
from __future__ import annotations
import os, re, json
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from unidecode import unidecode

from services.data_loader import load_tour_list, load_tour_detail

# Đường dẫn file phản hồi
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEEDBACK_FILE = os.path.join(os.path.dirname(BASE_DIR), "user_feedback_log.csv")

# Load danh sách tour
tour_df: pd.DataFrame = load_tour_list()

# Từ khóa phân loại tour theo chủ đề
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    # Biển / Sea
    "bien": [
        "bien", "bai", "vinh", "dao", "hon", "hai", "cang", "cap treo", "bo bien",
        "island", "beach", "vịnh", "bãi", "đảo", "hòn", "hải đảo", "cáp treo", "venice",
        "thi truong bien", "tam bien", "tam bien", "bo cat", "nước biển"
    ],
    # Núi / Mountain
    "nui": [
        "nui", "cao nguyen", "doi", "deo", "mang den", "ta dung", "pleiku",
        "kontum", "kon tum", "banah", "fansipan", "langbiang", "yok don",
        "sapa", "tay nguyen", "nuoc non", "thac", "rung", "moc chau", "mù cang chải"
    ],
    # Thành phố / City
    "thanh_pho": [
        "thanh pho", "pho", "pho co", "do thi", "city", "khu pho", "urban",
        "vinwonders", "vinpearl", "grand world", "shopping", "cho", "thi tran",
        "quan", "quan trung tam", "ben cang", "cau", "san bay", "resort", "khu dan cu",
        "quang truong", "pho di bo"
    ],
    # Khám phá thiên nhiên / Nature exploration
    "thien_nhien": [
        "thien nhien", "thiên nhiên", "rung", "rừng", "vuon quoc gia", "vườn quốc gia",
        "national park", "dong vat", "động vật", "ho nuoc", "hồ nước", "vuon cay", "vườn cây",
        "thac nuoc", "thác nước", "hoang da", "hoang dã", "cảnh quan", "tu nhien", "tự nhiên",
        "nature", "leo nui", "leo núi", "hang dong", "hang động", "cave"
    ],
    # Nghỉ dưỡng / Relaxation and resort
    "nghi_duong": [
        "nghi duong", "nghỉ dưỡng", "spa", "resort", "thu gian", "thư giãn", "cao cap", "cao cấp",
        "duong", "dưỡng", "duong sinh", "dưỡng sinh", "wellness", "retreat", "la veranda", "sanctuary"
    ],
    # Chụp ảnh / Photography & check-in
    "chup_anh": [
        "chup anh", "chụp ảnh", "check in", "check-in", "song ao", "sống ảo", "view dep", "view đẹp",
        "quan cafe", "quán cà phê", "studio", "photo", "photography", "instagram", "background", "pose"
    ],
    # Phiêu lưu mạo hiểm / Adventure and extreme
    "phieu_luu": [
        "phieu luu", "phiêu lưu", "mao hiem", "mạo hiểm", "trekking", "leo nui", "leo núi",
        "paragliding", "du luon", "dù lượn", "rafting", "hang dong", "hang động", "cave",
        "du thuyen", "du thuyền", "cano", "zipline", "kayak", "bungee", "cliff"
    ],
    # Văn hóa địa phương / Local culture
    "van_hoa": [
        "van hoa", "văn hoá", "văn hóa", "le hoi", "lễ hội", "di san", "di sản", "di tích", "chua", "chùa",
        "den", "đền", "dinh", "đình", "pagoda", "temple", "dan toc", "dân tộc", "lang", "làng",
        "truyen thong", "truyền thống", "traditional", "culture"
    ],
    # Ẩm thực / Cuisine
    "am_thuc": [
        "am thuc", "ẩm thực", "dac san", "đặc sản", "an uong", "ăn uống", "food", "ẩm thực địa phương",
        "food tour", "mon an", "món ăn", "lau", "lẩu", "pho", "phở", "bun", "bún", "cafe", "cà phê", "coffee"
    ]
}

# Model Pydantic (nếu chạy với FastAPI)
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

# Làm sạch văn bản (thường dùng để so sánh từ)
def clean_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return unidecode(text).lower().strip()

# Chuyển chuỗi giá sang số nguyên
def parse_price(price_str: Any) -> int:
    if not isinstance(price_str, str):
        return 0
    cleaned = re.sub(r"[^0-9]", "", price_str)
    try:
        return int(cleaned)
    except ValueError:
        return 0

# Chuyển an toàn sang số thực
def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

# Lấy số ngày từ tên tour
def extract_duration_from_name(name: str) -> Optional[int]:
    if not isinstance(name, str):
        return None
    match = re.search(r"(\d+)n", name.lower())
    if match:
        try:
            return int(match.group(1))
        except:
            return None
    return None

# Lấy phần mô tả tổng quan tour
def _get_tour_overview(row: Dict[str, Any]) -> str:
    detail_file = row.get("tour_detail_file")
    if not detail_file:
        return ""
    try:
        detail = load_tour_detail(detail_file)
    except:
        return ""
    overview = detail.get("overview", "")
    return overview if isinstance(overview, str) else ""

# Gộp thông tin mô tả để tạo văn bản đầu vào
def _compose_content_text(row: Dict[str, Any]) -> str:
    parts = [row.get("tour_name", ""), row.get("ribbon", ""), row.get("region_guess", ""), _get_tour_overview(row)]
    combined = " ".join(filter(None, parts))
    return clean_text(combined)

# Dự đoán danh mục từ văn bản
def _guess_categories(content_text: str) -> List[str]:
    categories: List[str] = []
    if not content_text:
        return categories
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if clean_text(kw) in content_text:
                categories.append(cat)
                break
    return categories

# Xử lý TF-IDF
tour_df["content_text"] = tour_df.apply(_compose_content_text, axis=1)
_text_corpus: List[str] = tour_df["content_text"].tolist()
_tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000)
_tfidf_matrix = _tfidf_vectorizer.fit_transform(_text_corpus)

# Dự đoán chủ đề từ keyword
tour_df["categories"] = tour_df["content_text"].apply(_guess_categories)

# Huấn luyện mô hình dự đoán vùng
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
except:
    _region_classifier = None

# Dự đoán vùng cho mỗi tour
def _predict_region_for_row(row: pd.Series) -> Optional[str]:
    if _region_classifier is None:
        return row.get("region_guess")
    try:
        vec = _tfidf_vectorizer.transform([row.get("content_text", "")])
        pred = _region_classifier.predict(vec)
        return str(pred[0]) if len(pred) else row.get("region_guess")
    except:
        return row.get("region_guess")

tour_df["predicted_region"] = tour_df.apply(_predict_region_for_row, axis=1)

# Huấn luyện mô hình phân loại chủ đề
_category_classifier: Optional[LogisticRegression] = None
try:
    _cat_labels, _cat_indices = [], []
    for idx, row in tour_df.iterrows():
        cats = row.get("categories", []) or []
        if cats:
            _cat_labels.append(cats[0])
            _cat_indices.append(idx)
    if len(set(_cat_labels)) > 1:
        _cat_train_X = _tfidf_matrix[_cat_indices]
        _cat_train_y = np.array(_cat_labels)
        cc = LogisticRegression(max_iter=300, multi_class='ovr')
        cc.fit(_cat_train_X, _cat_train_y)
        _category_classifier = cc
except:
    _category_classifier = None

# Dự đoán chủ đề chính
def _predict_category_for_row(row: pd.Series) -> Optional[str]:
    if _category_classifier is None:
        cats = row.get("categories", []) or []
        return cats[0] if cats else None
    try:
        vec = _tfidf_vectorizer.transform([row.get("content_text", "")])
        pred = _category_classifier.predict(vec)
        return str(pred[0]) if len(pred) else None
    except:
        cats = row.get("categories", []) or []
        return cats[0] if cats else None

tour_df["predicted_category"] = tour_df.apply(_predict_category_for_row, axis=1)

# Các đặc trưng dùng cho gợi ý
_feature_names = [
    "similarity_score", "budget_score", "rating_norm",
    "location_score", "duration_score", "category_score"
]

# Huấn luyện mô hình từ dữ liệu feedback
_feedback_model: Optional[LogisticRegression] = None
def _load_feedback_and_train() -> None:
    global _feedback_model
    if not os.path.exists(FEEDBACK_FILE):
        return
    try:
        df = pd.read_csv(FEEDBACK_FILE)
    except:
        return
    if not set(_feature_names + ["liked"]).issubset(df.columns) or len(df) < 20:
        return
    X = df[_feature_names].values
    y = df["liked"].astype(int).values
    try:
        model = LogisticRegression(max_iter=200)
        model.fit(X, y)
        _feedback_model = model
    except:
        _feedback_model = None

_load_feedback_and_train()

# Trọng số mặc định nếu không có model
DEFAULT_WEIGHTS: Dict[str, float] = {
    "similarity_score": 0.35, "budget_score": 0.15, "rating_norm": 0.15,
    "location_score": 0.1, "duration_score": 0.1, "category_score": 0.15
}

# Tính vector đặc trưng cho 1 tour
def _compute_feature_vector(row: pd.Series, user_profile: Dict[str, Any], similarity: float) -> Dict[str, float]:
    features: Dict[str, float] = {}
    features["similarity_score"] = float(similarity)

    user_budget = user_profile.get("ngan_sach")
    if user_budget and user_budget > 0:
        price = parse_price(row.get("price", 0))
        features["budget_score"] = max(0.0, 1.0 - (user_budget - price) / user_budget) if price <= user_budget else 0.0
    else:
        features["budget_score"] = 0.5

    features["rating_norm"] = min(max(safe_float(row.get("score", 0)) / 10.0, 0.0), 1.0)

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
        predicted_cat = row.get("predicted_category")
        tour_cats: set[str] = set(row.get("categories", []) or [])
        if predicted_cat:
            tour_cats.add(predicted_cat)
        features["category_score"] = len(desired_cats & tour_cats) / len(desired_cats) if tour_cats else 0.0
    else:
        features["category_score"] = 0.5
    return features

# Tính điểm tổng hợp
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

# Hàm gợi ý tour
def recommend_for_user(user_profile: Dict[str, Any], top_n: int = 5) -> List[Dict[str, Any]]:
    interests: List[str] = user_profile.get("so_thich", []) or []
    interest_text = clean_text(" ".join(interests))
    user_vec = _tfidf_vectorizer.transform([interest_text]) if interest_text else np.zeros((1, _tfidf_matrix.shape[1]))
    try:
        similarities = cosine_similarity(user_vec, _tfidf_matrix)[0]
    except:
        similarities = np.zeros(len(tour_df))
    results: List[Dict[str, Any]] = []
    for idx, row in tour_df.iterrows():
        similarity = float(similarities[idx]) if idx < len(similarities) else 0.0
        features = _compute_feature_vector(row, user_profile, similarity)

        user_budget = user_profile.get("ngan_sach")
        if user_budget and parse_price(row.get("price", 0)) > user_budget:
            continue
        min_rating = user_profile.get("rating")
        if min_rating and safe_float(row.get("score", 0)) < min_rating:
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
