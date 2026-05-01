"""Synthetic fraud-shaped dataset for Day 1.

The shape mirrors what you'd see in the Kaggle credit card fraud dataset
(numeric features + binary label) so we can swap in the real data on Day 6
without changing the training or serving code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

FEATURE_NAMES = [
    "amount",
    "hour_of_day",
    "merchant_risk_score",
    "user_txn_count_24h",
    "avg_txn_amount_30d",
    "distance_from_home_km",
    "is_foreign_txn",
    "card_age_days",
]


def load_synthetic_fraud(
    n_samples: int = 10_000,
    fraud_rate: float = 0.05,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate a synthetic fraud dataset with realistic feature names.

    Notes:
        We use a 5% fraud rate (not the real ~0.17%) so a vanilla classifier
        actually learns something useful on Day 1. We'll handle real-world
        class imbalance on Day 6 when we swap in the Kaggle dataset.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=len(FEATURE_NAMES),
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        weights=[1 - fraud_rate, fraud_rate],
        flip_y=0.01,
        random_state=random_state,
    )

    # Make features look plausible (positive amounts, bounded scores, etc.)
    # rng = np.random.default_rng(random_state)
    X[:, 0] = np.abs(X[:, 0]) * 50 + 10  # amount
    X[:, 1] = (X[:, 1] * 4 + 12).clip(0, 23).astype(int)  # hour_of_day
    X[:, 2] = (X[:, 2] * 0.2 + 0.5).clip(0, 1)  # merchant_risk_score
    X[:, 3] = np.abs(X[:, 3] * 5 + 3).astype(int)  # user_txn_count_24h
    X[:, 4] = np.abs(X[:, 4] * 30 + 50)  # avg_txn_amount_30d
    X[:, 5] = np.abs(X[:, 5] * 100)  # distance_from_home_km
    X[:, 6] = (X[:, 6] > 0.5).astype(int)  # is_foreign_txn
    X[:, 7] = np.abs(X[:, 7] * 500 + 100).astype(int)  # card_age_days

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    return df, pd.Series(y, name="is_fraud")
