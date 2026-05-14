"""Pydantic v2 request/response schemas for the fraud detection API.

The MLflow model signature types all 8 features as `double`, including
is_foreign_txn (which is conceptually a 0/1 flag). We accept bool at the
API boundary for a cleaner contract and coerce to float before inference.
"""

from pydantic import BaseModel, ConfigDict, Field


class Transaction(BaseModel):
    """Single transaction matching the fraud-detector model signature."""

    model_config = ConfigDict(extra="forbid")  # reject unknown fields

    amount: float = Field(..., ge=0, description="Transaction amount")
    hour_of_day: float = Field(..., ge=0, le=23, description="Hour 0-23")
    merchant_risk_score: float = Field(..., ge=0, le=1)
    user_txn_count_24h: float = Field(..., ge=0)
    avg_txn_amount_30d: float = Field(..., ge=0)
    distance_from_home_km: float = Field(..., ge=0)
    is_foreign_txn: bool = Field(..., description="True if foreign transaction")
    card_age_days: float = Field(..., ge=0)

    def to_feature_dict(self) -> dict[str, float]:
        """Coerce to the float-only dict the model expects."""
        return {
            "amount": self.amount,
            "hour_of_day": self.hour_of_day,
            "merchant_risk_score": self.merchant_risk_score,
            "user_txn_count_24h": self.user_txn_count_24h,
            "avg_txn_amount_30d": self.avg_txn_amount_30d,
            "distance_from_home_km": self.distance_from_home_km,
            "is_foreign_txn": float(self.is_foreign_txn),
            "card_age_days": self.card_age_days,
        }


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction: int = Field(..., description="0 = legitimate, 1 = fraud")
    fraud_probability: float = Field(..., ge=0, le=1)
    model_version: str
    model_alias: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_name: str
    model_version: str
    model_alias: str
    mlflow_tracking_uri: str
