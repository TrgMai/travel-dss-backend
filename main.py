# main.py
from fastapi import FastAPI, APIRouter
from recommend_tours import recommend_for_user, RecommendRequest
from schedule_builder import build_schedule, ScheduleRequest
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_router = APIRouter()

@app.get("/")
def root():
    return {"message": "Hello from Tour API"}

@app.post("/recommend")
def recommend_endpoint(req: RecommendRequest):
    profile = req.dict()
    result = recommend_for_user(profile)
    return {"results": result}

@app.post("/build_schedule")
def get_schedule(req: ScheduleRequest):
    return build_schedule(req)

app.include_router(api_router, prefix="/api")
