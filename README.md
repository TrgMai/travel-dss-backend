# travel-dss-backend
# run server: 
```uvicorn main:app --host 0.0.0.0 --port 10000 ```
# run server: 
```python -m uvicorn main:app --host 0.0.0.0 --port 10000```

## 🧠 API Endpoints

### 🔍 `POST /api/recommend`
Gợi ý tour phù hợp dựa vào sở thích người dùng.

**Request:**
```json
{
  "ngan_sach": 6000000,
  "so_thich": ["biển", "hoàng hôn", "ẩm thực"],
  "location": "phu-quoc",
  "duration": 3,
  "rating": 8.0
}
```

---

### 📅 `POST /api/build_schedule`
Trả về thông tin tour + lịch trình + khách sạn theo yêu cầu.

**Request:**
```json
{
  "tour_id": "1049",
  "location": "phu-quoc",
  "ngan_sach": 6000000,
  "so_thich": ["biển", "cầu hôn", "check-in"],
  "duration": 3,
  "rating": 8.5
}
```

---

### 🏨 `POST /api/hotels/by-location`
Tìm khách sạn theo tên vùng (tên gần đúng, không cần dấu).

**Request:**
```json
{
  "location": "phuquoc"
}
```

---

### 🏨 `POST /api/hotels/by-name`
Tìm khách sạn theo tên gần đúng.

**Request:**
```json
{
  "name": "sunset"
}
```

---

### 🔥 `GET /api/tours/hot`
Trả về 10 tour hot nhất (sắp xếp theo điểm `score`).

---

## 📎 Swagger UI

Để thử API dễ dàng:  
👉 http://localhost:10000/docs

---


