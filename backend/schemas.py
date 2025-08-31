from pydantic import BaseModel
from typing import List, Dict, Any

class DetectRequest(BaseModel):
    stock: str
    pattern: str
    timeframe: str
    chart_type: str

class ChartResponse(BaseModel):
    timeframe: str
    figure: Dict[str, Any]  # Plotly JSON

class DetectResponse(BaseModel):
    charts: List[ChartResponse]
