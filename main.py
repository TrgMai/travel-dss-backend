from fastapi import FastAPI, Request
from recommend_tours import recommend_for_user
from user_profile import build_user_profile

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello from Tour API"}

@app.post("/recommend")
async def recommend_endpoint(req: Request):
    body = await req.json()
    user_profile = build_user_profile(body)
    recommendations = recommend_for_user(user_profile)
    return {"recommendations": recommendations}