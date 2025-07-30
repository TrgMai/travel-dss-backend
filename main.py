# main.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from services.recommend_tours import *
from services.schedule_builder import *
from services.hotel_service import *
from services.tour_service import *

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Hello from Tour API"}

@app.post("/api/recommend")
def recommend_endpoint(req: RecommendRequest):
    profile = req.dict()
    result = recommend_for_user(profile)
    return {"results": result}

@app.post("/api/build_schedule")
def get_schedule(req: ScheduleRequest):
    return build_schedule(req)

@app.post("/api/hotels/by-location")
def get_hotels(req: HotelSearchRequest):
    hotels, matched_region = get_hotels_by_fuzzy_location(req.location)
    if matched_region is None:
        return {
            "location": req.location,
            "message": "Không tìm thấy vùng phù hợp.",
            "hotels": []
        }

    return {
        "location": req.location,
        "matched_region": matched_region,
        "count": len(hotels),
        "hotels": hotels[:10]
    }

@app.post("/api/hotels/by-name")
def search_hotels(req: HotelSearchByNameRequest):
    hotels = search_hotels_by_name(req.name)
    return {
        "search_term": req.name,
        "count": len(hotels),
        "results": hotels[:10]
    }

@app.get("/api/tours/hot")
def get_hot_tours():
    top_tours = get_top_hot_tours()
    return {
        "count": len(top_tours),
        "tours": top_tours
    }