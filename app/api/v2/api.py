from fastapi import APIRouter
from .endpoints import table_extraction

api_router = APIRouter()
api_router.include_router(
    table_extraction.router,
    prefix="/tables",
    tags=["table-extraction"]
)
