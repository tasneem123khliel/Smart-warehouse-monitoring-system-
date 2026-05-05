"""
database.py
-----------
SQLAlchemy setup + all table definitions for Smart Warehouse.

Tables
──────
  employees        → registered faces (from Attendance system)
  attendance_logs  → login / logout events
  helmet_logs      → per-frame or per-interval results from the YOLO API
  violations       → only the violation frames (easy to query for reports)
"""

from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String,
    DateTime, Boolean, Float, Text, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# ── Change this string to switch between SQLite (dev) and PostgreSQL (prod) ──
# SQLite  (no server needed):
DATABASE_URL = "sqlite:///./smart_warehouse.db"

# PostgreSQL (uncomment when deploying):
# DATABASE_URL = "postgresql://user:password@localhost:5432/smart_warehouse"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}  # SQLite only — remove for Postgres
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


# ════════════════════════════════════════════════════════════════════
#  TABLE 1 — employees
# ════════════════════════════════════════════════════════════════════
class Employee(Base):
    __tablename__ = "employees"

    id              = Column(Integer, primary_key=True, index=True)
    name            = Column(String(100), unique=True, nullable=False, index=True)
    department      = Column(String(100), nullable=True)
    face_pickle_path = Column(String(255), nullable=True)   # path to .pickle file
    registered_at   = Column(DateTime, default=datetime.utcnow)
    is_active       = Column(Boolean, default=True)

    # relationships
    attendance_records = relationship("AttendanceLog", back_populates="employee")

    def __repr__(self):
        return f"<Employee id={self.id} name={self.name}>"


# ════════════════════════════════════════════════════════════════════
#  TABLE 2 — attendance_logs
# ════════════════════════════════════════════════════════════════════
class AttendanceLog(Base):
    __tablename__ = "attendance_logs"

    id          = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=True)
    employee_name = Column(String(100), nullable=False)     # denormalised for speed
    action      = Column(String(10), nullable=False)        # "in" or "out"
    timestamp   = Column(DateTime, default=datetime.utcnow, index=True)
    confidence  = Column(Float, nullable=True)              # face-match confidence (optional)
    source      = Column(String(50), default="camera")      # "camera" / "manual"

    employee = relationship("Employee", back_populates="attendance_records")

    def __repr__(self):
        return f"<AttendanceLog {self.employee_name} {self.action} @ {self.timestamp}>"


# ════════════════════════════════════════════════════════════════════
#  TABLE 3 — helmet_logs   (one row per API call / video interval)
# ════════════════════════════════════════════════════════════════════
class HelmetLog(Base):
    __tablename__ = "helmet_logs"

    id               = Column(Integer, primary_key=True, index=True)
    timestamp        = Column(DateTime, default=datetime.utcnow, index=True)
    video_filename   = Column(String(255), nullable=True)
    frame_number     = Column(Integer, nullable=True)       # which frame (if per-frame)
    persons_count    = Column(Integer, default=0)
    helmets_count    = Column(Integer, default=0)
    no_helmet_count  = Column(Integer, default=0)           # classes 2 & 3
    violations_count = Column(Integer, default=0)
    status           = Column(String(20), default="SAFE")   # "SAFE" or "VIOLATION"
    confidence_avg   = Column(Float, nullable=True)         # avg YOLO confidence
    output_video_path = Column(String(255), nullable=True)  # saved annotated video

    violations = relationship("Violation", back_populates="helmet_log")

    def __repr__(self):
        return f"<HelmetLog id={self.id} status={self.status} violations={self.violations_count}>"


# ════════════════════════════════════════════════════════════════════
#  TABLE 4 — violations   (detailed per-detection row)
# ════════════════════════════════════════════════════════════════════
class Violation(Base):
    __tablename__ = "violations"

    id             = Column(Integer, primary_key=True, index=True)
    helmet_log_id  = Column(Integer, ForeignKey("helmet_logs.id"), nullable=False)
    timestamp      = Column(DateTime, default=datetime.utcnow)
    class_id       = Column(Integer)                        # 0=person,2=no_helmet,3=head
    class_name     = Column(String(50))
    confidence     = Column(Float)
    bbox_x1        = Column(Float)
    bbox_y1        = Column(Float)
    bbox_x2        = Column(Float)
    bbox_y2        = Column(Float)
    frame_number   = Column(Integer, nullable=True)

    helmet_log = relationship("HelmetLog", back_populates="violations")

    def __repr__(self):
        return f"<Violation id={self.id} class={self.class_name} conf={self.confidence:.2f}>"


# ════════════════════════════════════════════════════════════════════
#  DB initialiser — call once on startup
# ════════════════════════════════════════════════════════════════════
def init_db():
    Base.metadata.create_all(bind=engine)
    print("[DB] Tables created / verified ✓")


# ── Dependency for FastAPI ──────────────────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
