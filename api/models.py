"""
Pydantic models for API request/response schemas
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, date
from enum import Enum

class PatternType(str, Enum):
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    CUP_AND_HANDLE = "cup_and_handle"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"

class DetectionMode(str, Enum):
    STRICT = "strict"
    LENIENT = "lenient"
    BOTH = "both"

class SwingMethod(str, Enum):
    ROLLING = "rolling"
    ZIGZAG = "zigzag"
    FRACTAL = "fractal"

class ChartType(str, Enum):
    CANDLE = "candle"
    LINE = "line"
    OHLC = "ohlc"

class Backend(str, Enum):
    PLOTLY = "plotly"
    MATPLOTLIB = "matplotlib"

# Request Models
class PatternDetectionRequest(BaseModel):
    symbols: Union[List[str], str] = Field(default="random", description="List of symbols or 'random'")
    patterns: Union[List[PatternType], str] = Field(default="all", description="List of patterns or 'all'")
    timeframes: List[str] = Field(default=["1y", "2y", "3y"], description="List of timeframes")
    mode: DetectionMode = Field(default=DetectionMode.STRICT, description="Detection mode")
    swing_method: SwingMethod = Field(default=SwingMethod.ROLLING, description="Swing detection method")
    require_preceding_trend: bool = Field(default=True, description="Require preceding trend")
    max_patterns_per_timeframe: int = Field(default=10, ge=1, le=50, description="Max patterns per timeframe")
    min_patterns: int = Field(default=0, ge=0, description="Minimum patterns required")
    random_count: Optional[int] = Field(default=None, ge=1, le=100, description="Number of random symbols")

class SingleSymbolDetectionRequest(BaseModel):
    symbol: str = Field(description="Stock symbol")
    patterns: Union[List[PatternType], str] = Field(default="all", description="List of patterns or 'all'")
    timeframes: List[str] = Field(default=["1y", "2y", "3y"], description="List of timeframes")
    mode: DetectionMode = Field(default=DetectionMode.STRICT, description="Detection mode")
    swing_method: SwingMethod = Field(default=SwingMethod.ROLLING, description="Swing detection method")
    require_preceding_trend: bool = Field(default=True, description="Require preceding trend")

class PatternValidationRequest(BaseModel):
    pattern: Dict[str, Any] = Field(description="Pattern data to validate")
    symbol: str = Field(description="Stock symbol")
    timeframe: str = Field(description="Timeframe")

class BatchValidationRequest(BaseModel):
    patterns: List[Dict[str, Any]] = Field(description="List of patterns to validate")

class StockDataRequest(BaseModel):
    symbols: List[str] = Field(description="List of stock symbols", min_items=1)
    period: str = Field(default="1y", description="Data period")
    interval: str = Field(default="1d", description="Data interval")

class ChartGenerationRequest(BaseModel):
    symbol: str = Field(description="Stock symbol")
    timeframe: str = Field(description="Timeframe")
    pattern: Optional[Dict[str, Any]] = Field(default=None, description="Pattern to highlight")
    chart_type: ChartType = Field(default=ChartType.CANDLE, description="Chart type")
    backend: Backend = Field(default=Backend.PLOTLY, description="Plotting backend")
    width: Optional[int] = Field(default=1200, ge=400, le=2000, description="Chart width")
    height: Optional[int] = Field(default=800, ge=300, le=1500, description="Chart height")

class ComparisonChartRequest(BaseModel):
    symbols: List[str] = Field(description="Symbols to compare", min_items=2, max_items=5)
    timeframe: str = Field(description="Timeframe")
    chart_type: ChartType = Field(default=ChartType.LINE, description="Chart type")
    normalize: bool = Field(default=False, description="Normalize prices")

class ReportGenerationRequest(BaseModel):
    symbols: Optional[List[str]] = Field(default=None, description="Symbols to include")
    date_from: Optional[date] = Field(default=None, description="Start date")
    date_to: Optional[date] = Field(default=None, description="End date") 
    pattern_types: Optional[List[PatternType]] = Field(default=None, description="Pattern types to include")
    format: str = Field(default="csv", description="Report format: csv, json, pdf")
    include_charts: bool = Field(default=False, description="Include charts in report")
    min_score: Optional[int] = Field(default=None, ge=0, le=10, description="Minimum validation score")

# Response Models
class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: int = Field(ge=0, le=100)
    message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class PatternPoint(BaseModel):
    date: Union[str, datetime]
    price: float
    index: int

class DetectedPattern(BaseModel):
    type: PatternType
    symbol: str
    timeframe: str
    confidence: float = Field(ge=0, le=1)
    points: Dict[str, PatternPoint]
    neckline_level: Optional[float] = None
    breakout: Optional[PatternPoint] = None
    volume_confirmation: bool = False
    preceding_trend: Optional[str] = None

class ValidationResult(BaseModel):
    score: int = Field(ge=0, le=10)
    is_valid: bool
    details: Optional[Dict[str, Any]] = None
    confidence: float = Field(ge=0, le=1)

class PatternResult(BaseModel):
    symbol: str
    timeframe: str
    pattern_type: PatternType
    pattern_data: DetectedPattern
    validation: Optional[ValidationResult] = None
    chart_url: Optional[str] = None
    detected_at: datetime = Field(default_factory=datetime.now)

class StockData(BaseModel):
    symbol: str
    period: str
    interval: str
    data: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None

class ChartResponse(BaseModel):
    chart_id: str
    chart_url: str
    symbol: str
    timeframe: str
    chart_type: ChartType
    backend: Backend
    created_at: datetime = Field(default_factory=datetime.now)

class ReportResponse(BaseModel):
    report_id: str
    format: str
    download_url: str
    created_at: datetime = Field(default_factory=datetime.now)
    symbols_count: int
    patterns_count: int

class ScoreStatistics(BaseModel):
    pattern_type: PatternType
    total_patterns: int
    average_score: float
    valid_patterns: int
    validation_rate: float
    score_distribution: Dict[str, int]

class SystemStatus(BaseModel):
    status: str
    version: str
    uptime: str
    memory_usage: Optional[str] = None
    active_tasks: int
    total_patterns_detected: int
    last_detection: Optional[datetime] = None

class ErrorResponse(BaseModel):
    error: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)

class APIResponse(BaseModel):
    data: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

# Configuration Models
class PatternConfig(BaseModel):
    min_swing_threshold: float = Field(ge=0, le=1)
    min_pattern_strength: float = Field(ge=0, le=1)
    volume_threshold: float = Field(ge=0, le=10)
    preceding_trend_periods: int = Field(ge=5, le=100)

class HNSConfig(PatternConfig):
    min_head_prominence: float = Field(default=0.05, ge=0, le=1)
    max_shoulder_asymmetry: float = Field(default=0.25, ge=0, le=1)
    max_neckline_slope: float = Field(default=0.02, ge=0, le=1)

class CupHandleConfig(PatternConfig):
    min_cup_depth: float = Field(default=0.15, ge=0, le=1)
    max_cup_depth: float = Field(default=0.5, ge=0, le=1)
    max_handle_depth: float = Field(default=0.15, ge=0, le=1)
    rim_tolerance: float = Field(default=0.05, ge=0, le=1)

class DoublePatternConfig(PatternConfig):
    peak_similarity_threshold: float = Field(default=0.05, ge=0, le=1)
    min_valley_depth: float = Field(default=0.1, ge=0, le=1)
    time_symmetry_tolerance: float = Field(default=0.5, ge=0, le=2)

# File Upload Models
class UploadResponse(BaseModel):
    file_id: str
    filename: str
    size: int
    upload_time: datetime = Field(default_factory=datetime.now)
    status: str

class FileInfo(BaseModel):
    file_id: str
    filename: str
    size: int
    created_at: datetime
    file_type: str
    description: Optional[str] = None
