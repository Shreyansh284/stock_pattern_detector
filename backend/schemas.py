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
    mode: Optional[str] = None        # 'lenient' | 'strict'

class ChartResponse(BaseModel):
    timeframe: str
    figure: Dict[str, Any]  # Plotly JSON

class DetectResponse(BaseModel):
    charts: List[ChartResponse]
    
class DetectAllRequest(BaseModel):
    """Request payload for running detection across all stocks."""
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD

class ChartHtml(BaseModel):
    """HTML chart data for a specific timeframe."""
    timeframe: str
    pattern: str
    html: str

class StockPatternResult(BaseModel):
    """Pattern detection result for a single stock."""
    stock: str
    patterns: List[str]
    pattern_counts: Dict[str, int]
    current_price: float
    current_volume: int
    count: int
    charts: List[ChartHtml]

class DetectAllResponse(BaseModel):
    """Response containing detection results across all stocks."""
    results: List[StockPatternResult]
