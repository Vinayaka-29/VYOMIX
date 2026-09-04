from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router as api_router, UPLOAD_BASE_DIR
from models.model_server import model_server


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure upload directory exists
    UPLOAD_BASE_DIR.mkdir(parents=True, exist_ok=True)
    yield
    # Shutdown logic if needed


app = FastAPI(
    title="SatQuery AI - Backend API",
    description="Backend service for SatQuery AI (SIH 2026, Problem Statement 26167 - ISRO/SAC, Team Vyomix)",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend development and local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"  # Allow wildcard for local hackathon testing
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(api_router, prefix="")


@app.get("/health")
async def health_check():
    """
    Basic health-check endpoint returning operational status.
    """
    return {
        "status": "ok",
        "service": "SatQuery AI Backend",
        "problem_statement": "26167 (ISRO/SAC)",
        "team": "Vyomix",
        "version": "1.0.0",
        "model": model_server.status(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
