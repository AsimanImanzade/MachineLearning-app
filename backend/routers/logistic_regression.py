"""Logistic Regression router."""
from fastapi import APIRouter
from models.schemas import ClassificationRequest
from services.datasets import get_classification_dataset
from services.ml_engine import train_classification, get_decision_boundary

router = APIRouter(prefix="/api/logistic", tags=["Logistic Regression"])


@router.post("/train")
def train_logistic(req: ClassificationRequest):
    res = train_classification(
        *get_classification_dataset(req.dataset, req.noise, random_state=req.random_state),
        model_type="logistic",
        C=req.C,
        penalty=req.penalty,
        test_size=req.test_size,
        random_state=req.random_state,
    )
    boundary = get_decision_boundary(
        res["model"], res["scaler"],
        res["X_train"], res["X_test"],
        res["y_train"], res["y_test"],
    )
    return {
        "metrics": {
            "accuracy": res["accuracy"],
            "precision": res["precision"],
            "recall": res["recall"],
            "f1": res["f1"],
            "confusion_matrix": res["confusion_matrix"],
        },
        "boundary": boundary,
    }
