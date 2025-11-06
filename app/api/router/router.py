from fastapi import APIRouter
from app.api.Controller.DocumentController import router as document_router

# Create the main API router
api_router = APIRouter()

# Include all individual routers
api_router.include_router(document_router, tags=["Document Processing"])
