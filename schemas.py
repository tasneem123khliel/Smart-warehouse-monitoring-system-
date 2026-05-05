"""
schemas.py
----------
Pydantic models used by FastAPI for request validation and response serialisation.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


# ════════════════════════════════════════════════════════════════════
#  Employee schemas
# ════════════════════════════════════════════════════════════════════
class EmployeeCreate(BaseModel):
    name: str
    department: Optional[str] = None

class EmployeeOut(BaseModel):
    id: int
    name: str
    department: Optional[str]
    registered_at: datetime
    is_active: bool

    class Config:
        from_attributes = True


# ════════════════════════════════════════════════════════════════════
#  Attendance schemas
# ════════════════════════════════════════════════════════════════════
class AttendanceCreate(BaseModel):
    employee_name: str
    action: str = Field(..., pattern="^(in|out)$")
    confidence: Optional[float] = None
    source: str = "camera"

class AttendanceOut(BaseModel):
    id: int
    employee_name: str
    action: str
    timestamp: datetime
    confidence: Optional[float]

    class Config:
        from_attributes = True


# ════════════════════════════════════════════════════════════════════
#  Helmet detection schemas
# ════════════════════════════════════════════════════════════════════
class DetectionBox(BaseModel):
    class_id:   int
    class_name: str
    confidence: float
    bbox:       List[float]     # [x1, y1, x2, y2]
    is_violation: bool

class FrameResult(BaseModel):
    frame_number:     int
    persons_count:    int
    helmets_count:    int
    no_helmet_count:  int
    violations_count: int
    status:           str       # "SAFE" or "VIOLATION"
    detections:       List[DetectionBox]

class HelmetDetectionResponse(BaseModel):
    """Returned by POST /detect/helmet"""
    log_id:            int
    video_filename:    str
    total_frames:      int
    total_persons:     int
    total_helmets:     int
    total_violations:  int
    compliance_rate:   float    # helmets / persons * 100
    overall_status:    str
    output_video_path: Optional[str]
    frame_results:     List[FrameResult]

class HelmetLogOut(BaseModel):
    id:               int
    timestamp:        datetime
    video_filename:   Optional[str]
    persons_count:    int
    helmets_count:    int
    violations_count: int
    status:           str
    compliance_rate:  Optional[float] = None

    class Config:
        from_attributes = True


# ════════════════════════════════════════════════════════════════════
#  Summary / stats schemas
# ════════════════════════════════════════════════════════════════════
class AttendanceSummary(BaseModel):
    date:           str
    total_in:       int
    total_out:      int
    unique_employees: int

class SafetySummary(BaseModel):
    date:              str
    total_checks:      int
    total_violations:  int
    compliance_rate:   float
    worst_hour:        Optional[str]
