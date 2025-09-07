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
    data_source: Optional[str] = None # 'live' | 'past'
    stock_data_dir: Optional[str] = None

class ChartResponse(BaseModel):
    timeframe: str
    figure: Dict[str, Any]  # Plotly JSON

class DetectResponse(BaseModel):
    charts: List[ChartResponse]
    
class DetectAllRequest(BaseModel):
    """Request payload for running detection across all stocks."""
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD
    chart_type: Optional[str] = None  # 'candle' | 'line' | 'ohlc'
    data_source: Optional[str] = None # 'live' | 'past'
    stock_data_dir: Optional[str] = None
    stock_limit: Optional[int] = None  # optional limit for number of stocks when data_source='past'

class ChartHtml(BaseModel):
    """HTML chart data for a specific timeframe."""
    timeframe: str
    pattern: str
    html: str
    strength: Optional[str] = None  # 'strong' | 'weak'
    explanation: Optional[Dict[str, Any]] = None

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
