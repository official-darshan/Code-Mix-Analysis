from __future__ import annotations

from pathlib import Path
import logging
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

from src.predict import analyze_message
from database.database import init_db, save_analysis, fetch_all

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("support_api")

app = FastAPI(
    title="Code-Mixed Support Analytics API",
    description=(
        "NLP and ML API for support intent, sentiment, "
        "resolution and escalation prediction."
    ),
    version="3.0.0",
)

init_db()


class MessageRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Customer support message",
    )
    agent_response: str = Field(
        default="",
        max_length=2000,
        description=(
            "Optional agent response. Providing it improves resolution prediction."
        ),
    )


@app.get("/")
def root():
    return {
        "project": "Code-Mixed Support Conversation Analytics",
        "status": "running",
        "version": "3.0",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyze")
def analyze(request: MessageRequest):
    try:
        result = analyze_message(
            request.message,
            request.agent_response,
        )
        save_analysis(
            request.message,
            result,
            request.agent_response,
        )
        return {
            "customer_message": request.message,
            "agent_response": request.agent_response,
            **result,
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    except Exception:
        logger.exception("Unexpected prediction error")
        raise HTTPException(
            status_code=500,
            detail="Unable to analyze this message right now. Please try again.",
        )


@app.get("/analytics")
def analytics():
    try:
        return {"records": fetch_all()}
    except Exception:
        logger.exception("Analytics database error")
        raise HTTPException(
            status_code=500,
            detail="Unable to load analytics data.",
        )
