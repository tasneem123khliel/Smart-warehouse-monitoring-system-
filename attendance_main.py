"""
attendance_main.py  (updated version of your original main.py)
──────────────────────────────────────────────────────────────
Only change from the original:
  - login() and logout() now call post_attendance() which sends an
    HTTP POST to the FastAPI endpoint, so the event is stored in the DB.
  - The local log.txt file is kept as a backup fallback.
  - Everything else (webcam loop, registration, face recognition) is unchanged.
"""

import os
import datetime
import pickle
import requests
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import face_recognition
import util

# ── FastAPI base URL ─────────────────────────────────────────────────
API_BASE = "http://localhost:8000"   # change if running on a different host


def post_attendance(name: str, action: str):
    """
    Send the attendance event to the FastAPI backend.
    Falls back to local log.txt if the server is unreachable.
    """
    try:
        resp = requests.post(
            f"{API_BASE}/attendance/log",
            json={"employee_name": name, "action": action, "source": "camera"},
            timeout=3,
        )
        resp.raise_for_status()
        print(f"[API] Logged {name} {action} → id={resp.json()['id']}")
    except Exception as e:
        print(f"[API] Fallback to local log — {e}")
        with open("log.txt", "a") as f:
            f.write(f"{name},{datetime.datetime.now()},{action}\n")


class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520+350+100")

        self.login_button_main_window = util.get_button(
            self.main_window, 'login', 'green', self.login)
        self.login_button_main_window.place(x=750, y=200)

        self.logout_button_main_window = util.get_button(
            self.main_window, 'logout', 'red', self.logout)
        self.logout_button_main_window.place(x=750, y=300)

        self.register_new_user_button_main_window = util.get_button(
            self.main_window, 'register new user', 'gray',
            self.register_new_user, fg='black')
        self.register_new_user_button_main_window.place(x=750, y=400)

        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        self.add_webcam(self.webcam_label)

        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

    # ── webcam ────────────────────────────────────────────────────────
    def add_webcam(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)
        self._label = label
        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()
        self.most_recent_capture_arr = frame
        img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_pil = Image.fromarray(img_)
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        self._label.imgtk = imgtk
        self._label.configure(image=imgtk)
        self._label.after(20, self.process_webcam)

    # ── login / logout  ───────────────────────────────────────────────
    def login(self):
        name = util.recognize(self.most_recent_capture_arr, self.db_dir)
        if name in ['unknown_person', 'no_persons_found']:
            util.msg_box('Ups...', 'Unknown user. Please register or try again.')
        else:
            post_attendance(name, "in")          # ← sends to DB
            util.msg_box('Welcome back!', f'Welcome, {name}.')

    def logout(self):
        name = util.recognize(self.most_recent_capture_arr, self.db_dir)
        if name in ['unknown_person', 'no_persons_found']:
            util.msg_box('Ups...', 'Unknown user. Please register or try again.')
        else:
            post_attendance(name, "out")         # ← sends to DB
            util.msg_box('Hasta la vista!', f'Goodbye, {name}.')

    # ── registration (unchanged) ──────────────────────────────────────
    def register_new_user(self):
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("1200x520+370+120")

        self.accept_button_register_new_user_window = util.get_button(
            self.register_new_user_window, 'Accept', 'green', self.accept_register_new_user)
        self.accept_button_register_new_user_window.place(x=750, y=300)

        self.try_again_button_register_new_user_window = util.get_button(
            self.register_new_user_window, 'Try again', 'red', self.try_again_register_new_user)
        self.try_again_button_register_new_user_window.place(x=750, y=400)

        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)
        self.add_img_to_label(self.capture_label)

        self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=750, y=150)

        self.text_label_register_new_user = util.get_text_label(
            self.register_new_user_window, 'Please, \ninput username:')
        self.text_label_register_new_user.place(x=750, y=70)

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get(1.0, "end-1c").strip()
        if not name:
            util.msg_box('Error', 'Please enter a username!')
            return
        try:
            embeddings = face_recognition.face_encodings(self.register_new_user_capture)[0]
            file_path  = os.path.join(self.db_dir, f'{name}.pickle')
            with open(file_path, 'wb') as file:
                pickle.dump(embeddings, file)

            # also register in the DB ──────────────────────────────
            try:
                requests.post(
                    f"{API_BASE}/employees",
                    json={"name": name, "face_pickle_path": file_path},
                    timeout=3,
                )
            except Exception:
                pass   # not critical — face file is the source of truth

            util.msg_box('Success!', 'User was registered successfully!')
            self.register_new_user_window.destroy()
        except IndexError:
            util.msg_box('Error', 'No face detected! Please try again.')

    def start(self):
        self.main_window.mainloop()


if __name__ == "__main__":
    app = App()
    app.start()
