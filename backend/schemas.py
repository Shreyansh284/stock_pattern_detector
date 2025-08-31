from pydantic import BaseModel
from typing import List, Dict, Any

class DetectRequest(BaseModel):
    stock: str
    pattern: str
    timeframes: List[str]

class ChartResponse(BaseModel):
    timeframe: str
    figure: Dict[str, Any]  # Plotly JSON

class DetectResponse(BaseModel):
    charts: List[ChartResponse]
