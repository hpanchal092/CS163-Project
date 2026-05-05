from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Product Success Prediction API")

class ProductFeatures(BaseModel):
    n_reviews_4w: float
    mean_rating_4w: float
    reviews_per_day_4w: float
    sum_vote_4w: float

@app.get("/")
def home():
    return {"message": "Cloud Run inference service is running"}

@app.post("/predict")
def predict(features: ProductFeatures):
    score = (
        0.35 * min(features.n_reviews_4w / 20, 1)
        + 0.20 * min(features.mean_rating_4w / 5, 1)
        + 0.25 * min(features.reviews_per_day_4w / 1, 1)
        + 0.20 * min(features.sum_vote_4w / 50, 1)
    )

    return {
        "success_probability": round(score, 3),
        "prediction": "High potential" if score >= 0.5 else "Low potential"
    }
