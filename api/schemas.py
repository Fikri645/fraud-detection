"""Pydantic request/response models for the scoring API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    """A single raw transaction, as it would arrive from a payment gateway."""
    cc_num: int = Field(..., description="Card number (identifier)")
    amt: float = Field(..., ge=0, description="Transaction amount")
    unix_time: float = Field(..., description="Transaction time (unix seconds)")
    merchant: str = ""
    category: str = ""
    gender: str = ""
    state: str = ""
    lat: float = 0.0
    long: float = 0.0
    merch_lat: float = 0.0
    merch_long: float = 0.0
    city_pop: float = 0.0
    dob: str = "1980-01-01"

    model_config = {
        "json_schema_extra": {
            "example": {
                "cc_num": 1234567890123456, "amt": 842.15,
                "unix_time": 1593561600, "merchant": "fraud_Kozey-Boehm",
                "category": "shopping_net", "gender": "F", "state": "NY",
                "lat": 40.71, "long": -74.0, "merch_lat": 36.0, "merch_long": -90.0,
                "city_pop": 1000000, "dob": "1988-03-09",
            }
        }
    }


class FeatureContribution(BaseModel):
    feature: str
    contribution: float


class ScoreResponse(BaseModel):
    fraud_probability: float
    decision: str               # "approve" | "review" | "decline"
    threshold: float
    latency_ms: float
    top_factors: list[FeatureContribution]
