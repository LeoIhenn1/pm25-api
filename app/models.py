from pydantic import BaseModel
from typing import Optional

class DataEntry(BaseModel):
    Latitude: float
    Longitude: float
    PM2_5: float  # Adjusted field name for Pydantic

class DataEntryResponse(BaseModel):
    message: str
    id: int
