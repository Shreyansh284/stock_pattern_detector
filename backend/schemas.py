from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class DetectRequest(BaseModel):
    stock: str
    pattern: str
    chart_type: str
    # Either provide timeframe OR start_date+end_date
    timeframe: Optional[str] = None
    start_date: Optional[str] = None  # YYYY-MM-DD
    end_date: Optional[str] = None    # YYYY-MM-DD

class ChartResponse(BaseModel):
    timeframe: str
    figure: Dict[str, Any]  # Plotly JSON

class DetectResponse(BaseModel):
    charts: List[ChartResponse]
