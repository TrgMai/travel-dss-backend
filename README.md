# travel-dss-backend
# run server: 
```uvicorn main:app --host 0.0.0.0 --port 10000 ```
# run server: 
```python -m uvicorn main:app --host 0.0.0.0 --port 10000```

# api/recommend
```json
{
  "ngan_sach": 6000000,
  "so_thich": ["biển", "hoàng hôn", "ẩm thực"],
  "location": "phu-quoc",
  "duration": 3,
  "rating": 8.0
}
```
# api/build_schedule
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

