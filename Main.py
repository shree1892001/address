import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.Common.Constants import *
from app.api.router.router import api_router
from app.api.v2.api import api_router as v2_router

# Create the main FastAPI app
app1 = FastAPI(
    title="OCR Extract Service",
    description="Main service with address lookup, autocomplete, search functionality, and document table extraction",
    version="1.0.0"
)

# Add comprehensive CORS middleware
app1.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,  # Cache preflight requests for 24 hours
)

# Include the API routers
app1.include_router(api_router, prefix="/api/v1")
app1.include_router(v2_router, prefix="/api/v2")

if __name__ == "__main__":
    uvicorn.run(
        "Main:app1",
        host= API_HOST,
        port=API_PORT,
        reload=True
    )
