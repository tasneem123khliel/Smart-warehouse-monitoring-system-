"""
api.py - Smart Warehouse Monitoring System
"""

import os
import shutil
import pickle
import numpy as np
import cv2
from datetime import datetime, date
from typing import List, Optional

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func

from database import init_db, get_db, Employee, AttendanceLog, HelmetLog, Violation
from schemas import (
    EmployeeCreate, EmployeeOut,
    AttendanceCreate, AttendanceOut,
    HelmetDetectionResponse, HelmetLogOut,
)
from helmet_service import run_helmet_detection
import util

app = FastAPI(
    title="Smart Warehouse Monitoring API",
    description="Attendance tracking + PPE helmet compliance",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── paths ──────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(__file__)
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
UPLOAD_DIR   = os.path.join(BASE_DIR, "uploaded_videos")
OUTPUT_DIR   = os.path.join(BASE_DIR, "annotated_outputs")
FACES_DIR    = os.path.join(BASE_DIR, "db")           # face pickle files
MODEL_PATH   = os.path.join(BASE_DIR, "best.pt")

for d in [UPLOAD_DIR, OUTPUT_DIR, FACES_DIR]:
    os.makedirs(d, exist_ok=True)

# Serve static frontend
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.on_event("startup")
def on_startup():
    init_db()


# ════════════════════════════════════════════════════════════
#  Frontend
# ════════════════════════════════════════════════════════════
@app.get("/", tags=["Frontend"])
async def serve_dashboard():
    path = os.path.join(FRONTEND_DIR, "dashboard.html")
    if os.path.exists(path):
        return FileResponse(path)
    # fallback: look in same folder as api.py
    path2 = os.path.join(BASE_DIR, "dashboard.html")
    if os.path.exists(path2):
        return FileResponse(path2)
    return {"status": "ok", "docs": "/docs"}


# ════════════════════════════════════════════════════════════
#  EMPLOYEES
# ════════════════════════════════════════════════════════════
@app.post("/employees", response_model=EmployeeOut, tags=["Employees"])
def create_employee(payload: EmployeeCreate, db: Session = Depends(get_db)):
    existing = db.query(Employee).filter(Employee.name == payload.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="الموظف مسجل مسبقاً")
    emp = Employee(name=payload.name, department=payload.department)
    db.add(emp)
    db.commit()
    db.refresh(emp)
    return emp


@app.get("/employees", response_model=List[EmployeeOut], tags=["Employees"])
def list_employees(active_only: bool = True, db: Session = Depends(get_db)):
    q = db.query(Employee)
    if active_only:
        q = q.filter(Employee.is_active == True)
    return q.order_by(Employee.name).all()


@app.post("/employees/register-face", tags=["Employees"])
async def register_face(
    file: UploadFile = File(...),
    employee_name: str = Form(...),
    db: Session = Depends(get_db),
):
    """
    يستقبل صورة من الداشبورد، يستخرج الـ face embedding
    ويحفظه كـ pickle في مجلد db/
    """
    import face_recognition as fr

    contents = await file.read()
    nparr    = np.frombuffer(contents, np.uint8)
    frame    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="صورة غير صالحة")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = fr.face_encodings(rgb)

    if not encodings:
        raise HTTPException(status_code=400, detail="لم يتم اكتشاف وجه في الصورة")

    pickle_path = os.path.join(FACES_DIR, f"{employee_name}.pickle")
    with open(pickle_path, "wb") as f:
        pickle.dump(encodings[0], f)

    # update DB with pickle path
    emp = db.query(Employee).filter(Employee.name == employee_name).first()
    if emp:
        emp.face_pickle_path = pickle_path
        db.commit()

    return {"status": "success", "message": f"تم تسجيل وجه {employee_name}"}


# ════════════════════════════════════════════════════════════
#  ATTENDANCE
# ════════════════════════════════════════════════════════════
@app.post("/attendance/log", response_model=AttendanceOut, tags=["Attendance"])
def log_attendance(payload: AttendanceCreate, db: Session = Depends(get_db)):
    emp = db.query(Employee).filter(Employee.name == payload.employee_name).first()
    record = AttendanceLog(
        employee_id   = emp.id if emp else None,
        employee_name = payload.employee_name,
        action        = payload.action,
        confidence    = payload.confidence,
        source        = payload.source,
        timestamp     = datetime.utcnow(),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@app.post("/attendance/recognize", tags=["Attendance"])
async def recognize_face(
    file:   UploadFile = File(...),
    action: str        = Form("in"),
    db:     Session    = Depends(get_db),
):
    """
    يستقبل صورة من كاميرا الويب، يتعرف على الوجه،
    ويسجل الحضور/الخروج في DB.
    """
    contents = await file.read()
    nparr    = np.frombuffer(contents, np.uint8)
    frame    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="صورة غير صالحة")

    name = util.recognize(frame, FACES_DIR)

    if name in ["unknown_person", "no_persons_found"]:
        return {
            "status":  "unknown",
            "name":    None,
            "message": "الشخص غير معروف — يرجى التسجيل أولاً",
        }

    # save to DB
    emp = db.query(Employee).filter(Employee.name == name).first()
    record = AttendanceLog(
        employee_id   = emp.id if emp else None,
        employee_name = name,
        action        = action,
        source        = "web_camera",
        timestamp     = datetime.utcnow(),
    )
    db.add(record)
    db.commit()

    return {"status": "success", "name": name, "action": action}


@app.get("/attendance/logs", response_model=List[AttendanceOut], tags=["Attendance"])
def get_attendance_logs(
    employee_name: Optional[str]  = None,
    action:        Optional[str]  = None,
    from_date:     Optional[date] = None,
    to_date:       Optional[date] = None,
    limit:         int            = Query(100, le=1000),
    db:            Session        = Depends(get_db),
):
    q = db.query(AttendanceLog)
    if employee_name:
        q = q.filter(AttendanceLog.employee_name.ilike(f"%{employee_name}%"))
    if action:
        q = q.filter(AttendanceLog.action == action)
    if from_date:
        q = q.filter(AttendanceLog.timestamp >= datetime.combine(from_date, datetime.min.time()))
    if to_date:
        q = q.filter(AttendanceLog.timestamp <= datetime.combine(to_date, datetime.max.time()))
    return q.order_by(AttendanceLog.timestamp.desc()).limit(limit).all()


@app.get("/attendance/today", tags=["Attendance"])
def today_attendance_summary(db: Session = Depends(get_db)):
    today_start = datetime.combine(date.today(), datetime.min.time())
    logs = db.query(AttendanceLog).filter(AttendanceLog.timestamp >= today_start).all()
    ins  = [l for l in logs if l.action == "in"]
    outs = [l for l in logs if l.action == "out"]
    return {
        "date":             str(date.today()),
        "total_in":         len(ins),
        "total_out":        len(outs),
        "unique_employees": len({l.employee_name for l in logs}),
        "currently_inside": len({l.employee_name for l in ins}) - len({l.employee_name for l in outs}),
    }


# ════════════════════════════════════════════════════════════
#  HELMET / PPE DETECTION
# ════════════════════════════════════════════════════════════
@app.post("/detect/helmet", tags=["Helmet Detection"])
async def detect_helmet(
    video:          UploadFile = File(...),
    conf_threshold: float      = Form(0.5),
    sample_every:   int        = Form(5),
    db:             Session    = Depends(get_db),
):
    save_path = os.path.join(UPLOAD_DIR, video.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    try:
        result = run_helmet_detection(
            video_path            = save_path,
            output_dir            = OUTPUT_DIR,
            model_path            = MODEL_PATH,
            conf_threshold        = conf_threshold,
            sample_every_n_frames = sample_every,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    helmet_log = HelmetLog(
        timestamp         = datetime.utcnow(),
        video_filename    = result["video_filename"],
        persons_count     = result["total_persons"],
        helmets_count     = result["total_helmets"],
        violations_count  = result["total_violations"],
        status            = result["overall_status"],
        output_video_path = result["output_video_path"],
    )
    db.add(helmet_log)
    db.flush()

    # ── save violation rows ─────────────────────────────────
    for frame in result.get("frame_results", []):
        if frame.get("violations_count", 0) == 0:
            continue

        persons_in_frame   = [d for d in frame["detections"] if d["class_id"] == 0]
        helmets_in_frame   = [d for d in frame["detections"] if d["class_id"] == 1]
        unmatched_persons  = persons_in_frame[len(helmets_in_frame):]
        explicit_viols     = [d for d in frame["detections"] if d["is_violation"]]
        all_viols          = unmatched_persons + explicit_viols

        for det in all_viols:
            db.add(Violation(
                helmet_log_id = helmet_log.id,
                timestamp     = datetime.utcnow(),
                class_id      = det["class_id"],
                class_name    = det["class_name"],
                confidence    = det["confidence"],
                bbox_x1       = det["bbox"][0],
                bbox_y1       = det["bbox"][1],
                bbox_x2       = det["bbox"][2],
                bbox_y2       = det["bbox"][3],
                frame_number  = frame["frame_number"],
            ))

    db.commit()
    db.refresh(helmet_log)

    return {
        "log_id":            helmet_log.id,
        "video_filename":    result["video_filename"],
        "total_frames":      result["total_frames"],
        "total_persons":     result["total_persons"],
        "total_helmets":     result["total_helmets"],
        "total_violations":  result["total_violations"],
        "compliance_rate":   result["compliance_rate"],
        "overall_status":    result["overall_status"],
        "output_video_path": result["output_video_path"],
    }


@app.get("/helmet/logs", response_model=List[HelmetLogOut], tags=["Helmet Detection"])
def get_helmet_logs(
    status:    Optional[str]  = None,
    from_date: Optional[date] = None,
    to_date:   Optional[date] = None,
    limit:     int            = Query(100, le=1000),
    db:        Session        = Depends(get_db),
):
    q = db.query(HelmetLog)
    if status:
        q = q.filter(HelmetLog.status == status.upper())
    logs = q.order_by(HelmetLog.timestamp.desc()).limit(limit).all()
    out = []
    for log in logs:
        d = HelmetLogOut.model_validate(log)
        d.compliance_rate = (
            round(log.helmets_count / log.persons_count * 100, 1)
            if log.persons_count > 0 else 100.0
        )
        out.append(d)
    return out


@app.get("/helmet/stats", tags=["Helmet Detection"])
def helmet_stats(db: Session = Depends(get_db)):
    total_logs      = db.query(HelmetLog).count()
    total_viols     = db.query(func.sum(HelmetLog.violations_count)).scalar() or 0
    total_persons   = db.query(func.sum(HelmetLog.persons_count)).scalar() or 0
    total_helmets   = db.query(func.sum(HelmetLog.helmets_count)).scalar() or 0
    violation_logs  = db.query(HelmetLog).filter(HelmetLog.status == "VIOLATION").count()
    return {
        "total_videos_processed": total_logs,
        "violation_videos":       violation_logs,
        "safe_videos":            total_logs - violation_logs,
        "total_persons_detected": total_persons,
        "total_helmets_detected": total_helmets,
        "total_violations":       total_viols,
        "overall_compliance_pct": round(total_helmets / total_persons * 100, 1) if total_persons > 0 else 100.0,
    }


@app.get("/violations", tags=["Helmet Detection"])
def list_violations(
    from_date:  Optional[date] = None,
    class_name: Optional[str]  = None,
    limit:      int            = Query(200, le=2000),
    db:         Session        = Depends(get_db),
):
    q = db.query(Violation)
    if from_date:
        q = q.filter(Violation.timestamp >= datetime.combine(from_date, datetime.min.time()))
    if class_name:
        q = q.filter(Violation.class_name == class_name)
    return q.order_by(Violation.timestamp.desc()).limit(limit).all()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
