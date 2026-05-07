# 🏭 Smart Warehouse Monitoring System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?style=for-the-badge&logo=fastapi)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange?style=for-the-badge)
![SQLite](https://img.shields.io/badge/SQLite-Database-lightblue?style=for-the-badge&logo=sqlite)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**An intelligent warehouse monitoring system — Face Recognition Attendance + PPE Helmet Compliance Detection**

[Screenshots](#-screenshots) • [Installation](#-installation) • [Usage](#-usage) • [API Docs](#-api-endpoints)

</div>

---

## 📋 Overview

**Smart Warehouse** is a full-stack intelligent monitoring system designed for warehouses and industrial facilities. It combines:

- 🎭 **Automatic Attendance Tracking** using Face Recognition
- 🪖 **PPE Helmet Compliance Detection** powered by YOLOv8
- 📊 **Interactive Web Dashboard** for real-time monitoring and reports
- 🗄️ **Persistent Database** storing all logs, records, and violations

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 👤 **Face Recognition Attendance** | Automatically identifies employees via webcam and logs check-in/check-out |
| 🪖 **Helmet Detection** | Analyzes video footage to detect employees not wearing safety helmets |
| 📹 **Video Analysis** | Processes uploaded videos and produces annotated output with detection overlays |
| 📊 **Real-time Statistics** | Displays compliance rate, violation count, and currently present employees |
| 🧑‍💼 **Employee Management** | Register new employees and link their face embeddings |
| 🗂️ **Attendance Logs** | Browse and filter attendance records by name, date, and action |
| 🔌 **REST API** | Full programmatic API ready for integration and extension |

---

## 🖼️ Screenshots

<table>
  <tr>
    <td align="center"><b>🏠 Attendance Registration</b></td>
    <td align="center"><b>🪖 Safety Gear Analysis</b></td>
  </tr>
  <tr>
    <td><img src="images/main.PNG" width="400"/></td>
    <td><img src="images/save.PNG" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><b>📋 Attendance History</b></td>
    <td align="center"><b>⚠️ Violation Detection</b></td>
  </tr>
  <tr>
    <td><img src="images/history.PNG" width="400"/></td>
    <td><img src="images/violation.PNG" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><b>✅ Check-in</b></td>
    <td align="center"><b>🚪 Check-out</b></td>
  </tr>
  <tr>
    <td><img src="images/login.PNG" width="400"/></td>
    <td><img src="images/logout.PNG" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><b>🧑‍💼 New Employee Registration</b></td>
    <td></td>
  </tr>
  <tr>
    <td><img src="images/new_user.PNG" width="400"/></td>
    <td></td>
  </tr>
</table>

> 💡 **Note:** Place all screenshot images inside a `images/` folder next to this README file.

---

## 🏗️ Project Structure

```
smart-warehouse/
│
├── api.py                  # FastAPI — main API entry point
├── attendance_main.py      # Tkinter desktop app for webcam-based attendance
├── helmet_service.py       # YOLOv8 helmet detection logic
├── database.py             # SQLAlchemy models and database setup
├── schemas.py              # Pydantic schemas for request/response validation
├── util.py                 # Helper functions (face recognition, UI widgets)
├── best.pt                 # Trained YOLOv8 model for helmet detection
├── requirements.txt        # Python dependencies
├── smart_warehouse.db      # SQLite database (auto-created on first run)
│
├── db/                     # Face embedding pickle files
├── uploaded_videos/        # Raw videos uploaded for analysis
├── annotated_outputs/      # Annotated output videos with detection overlays
└── frontend/               # Web dashboard (HTML/CSS/JS)
    └── dashboard.html
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.8 or higher
- Webcam (for attendance registration)
- OS: Windows / Linux / macOS

### Setup Steps

```bash
# 1. Clone the repository
git clone https://github.com/tasneem123khliel/smart-warehouse-monitoring-system.git
cd smart-warehouse

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install face recognition library
pip install face_recognition

# Windows users — if you encounter issues:
# pip install cmake dlib face_recognition
```

> ⚠️ **Important:** Make sure `best.pt` is present in the project root before running.

---

## 🚀 Usage

### 1. Start the API Server

```bash
python api.py
# or
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

- API available at: `http://localhost:8000`
- Interactive Swagger docs: `http://localhost:8000/docs`

### 2. Launch the Desktop Attendance App

```bash
python attendance_main.py
```

### 3. Open the Web Dashboard

Navigate to `http://localhost:8000` in your browser.

---

## 🗄️ Database

The system uses **SQLite** by default — no setup required.  
To switch to **PostgreSQL** for production, update `DATABASE_URL` in `database.py`:

```python
# database.py
DATABASE_URL = "postgresql://user:password@localhost:5432/smart_warehouse"
```

### Database Tables

| Table | Description |
|-------|-------------|
| `employees` | Registered employee profiles |
| `attendance_logs` | Check-in and check-out events |
| `helmet_logs` | Video analysis results per submission |
| `violations` | Detailed records of each detected violation |

---

## 📡 API Endpoints

### Employees
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/employees` | Register a new employee |
| `GET` | `/employees` | List all employees |
| `POST` | `/employees/register-face` | Upload and save face embedding |

### Attendance
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/attendance/recognize` | Recognize face and log attendance |
| `POST` | `/attendance/log` | Manually log an attendance event |
| `GET` | `/attendance/logs` | Retrieve attendance records (filterable) |
| `GET` | `/attendance/today` | Today's attendance summary |

### Helmet Detection
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/detect/helmet` | Upload and analyze a video |
| `GET` | `/helmet/logs` | Retrieve helmet detection logs |
| `GET` | `/helmet/stats` | Overall compliance statistics |
| `GET` | `/violations` | List all recorded violations |

---

## 🧠 Detection Model (YOLOv8)

The `best.pt` model is trained on 4 classes:

| ID | Class | Description |
|----|-------|-------------|
| 0 | `person` | A person detected in frame |
| 1 | `helmet` | Safety helmet being worn |
| 2 | `no_helmet` | Head without a helmet |
| 3 | `head` | Bare head (potential violation) |

### Compliance Logic

- ✅ **SAFE** — Compliance rate ≥ 85%
- ⚠️ **VIOLATION** — Compliance rate < 85%

---

## 📦 Dependencies

```
fastapi==0.111.0
uvicorn[standard]==0.29.0
python-multipart==0.0.9
sqlalchemy==2.0.30
ultralytics>=8.0.0
opencv-python>=4.8.0
face_recognition
requests>=2.31.0
pydantic>=2.0.0
Pillow
tkinter (built-in with Python)
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

Built by ❤️ **Tasneem Yasser** for safer, smarter warehouses.

> For questions or support, feel free to open an issue on GitHub.
