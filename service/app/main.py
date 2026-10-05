"""
FastAPI application entrypoint for ChainGuard service.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from service.app.routes import router

app = FastAPI(
    title="ChainGuard Inference & Registry Service",
    description="Zero-Leakage ML Model Serving & On-Chain Risk Score Publisher",
    version="1.0.0",
)

# Enable CORS for local DApp frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service.app.main:app", host="0.0.0.0", port=8000, reload=True)
