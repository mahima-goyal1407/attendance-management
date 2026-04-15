import tkinter as tk
from tkinter import messagebox
import cv2 as c
import face_recognition as fr
import pandas as pd
import datetime as dt
import os

# ---------------- STUDENT DATA ----------------
student_data = {
    "125110023": {"name": "Abhijeet", "password": "Abhijeet@123"},
    "125110012": {"name": "Kshitij", "password": "Kshitij@123"},
    "125110009": {"name": "Yash", "password": "Yash@123"},
    "125110057": {"name": "Yuvraj", "password": "Yuvraj@123"},
    "125110001": {"name": "Harshvardhan", "password": "Harshvardhan@123"},
    "125110045": {"name": "Abhishek", "password": "Abhijeet123"},
    "125110033": {"name": "Anshu", "password": "Anshu@123"},
    "125110055": {"name": "Balwant", "password": "Balwant@123"},
    "125110030": {"name": "Mahima", "password": "Mahima@123"},
    "125110050": {"name": "Nupur", "password": "Nupur@123"},
    "125110051": {"name": "Sahil", "password": "Sahil@123"},
    "125110028": {"name": "Tanvi", "password": "Tanvi@123"},
    "125110031": {"name": "Ujjwala", "password": "Ujjwala@123"},
    "125110013": {"name": "Raunak", "password": "Raunak@123"},
    "125110002": {"name": "Anand", "password": "Anand@123"},
    "125110039": {"name": "Satyam", "password": "Satyam@123"},
    "125310001": {"name": "Anjali", "password": "Anjali@123"},
    "125110014": {"name": "Divyanshu", "password": "Divyanshu@123"},
    "125101095": {"name": "Puneet", "password": "Puneet@123"}
}

# ---------------- GLOBAL ----------------
known_face_encodings = []
known_rollnumber = []
known_names = []

attendance_mark = set()

# ---------------- LOAD FACES ----------------
def add_known_face(image_path, name, roll):
    if not os.path.exists(image_path):
        return

    image = fr.load_image_file(image_path)
    encodings = fr.face_encodings(image)

    if encodings:
        known_face_encodings.append(encodings[0])
        known_names.append(name)
        known_rollnumber.append(roll)

def load_faces():
    known_face_encodings.clear()
    known_names.clear()
    known_rollnumber.clear()

    for roll, data in student_data.items():
        file = data["name"].lower() + ".jpeg"
        add_known_face(file, data["name"], roll)

# ---------------- ATTENDANCE ----------------
def mark_attendance(name, roll, subject):
    global attendance_mark

    if roll in attendance_mark:
        return

    attendance_mark.add(roll)

    file = subject + ".csv"

    if os.path.exists(file):
        df = pd.read_csv(file)
        if roll in df["rollnumber"].values:
            return

    now = dt.datetime.now()
    data = pd.DataFrame([[name, now, roll]], columns=["name", "date&time", "rollnumber"])

    if not os.path.exists(file):
        data.to_csv(file, index=False)
    else:
        data.to_csv(file, mode="a", header=False, index=False)

# ---------------- FACE RECOGNITION ----------------
def start_recognition():
    subject = entry_subject.get()

    if subject == "":
        messagebox.showerror("Error", "Enter Subject Name")
        return

    attendance_mark.clear()
    load_faces()

    video = c.VideoCapture(0)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        small = c.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb = c.cvtColor(small, c.COLOR_BGR2RGB)

        faces = fr.face_locations(rgb)
        encodings = fr.face_encodings(rgb, faces)

        for (top, right, bottom, left), face_encoding in zip(faces, encodings):

            distances = fr.face_distance(known_face_encodings, face_encoding)

            name = "Unknown"
            roll = None
            color = (0, 0, 255)

            if len(distances) > 0:
                best_index = distances.argmin()

                # STRICT MATCH (ANTI-PROXY)
                if distances[best_index] < 0.5:
                    name = known_names[best_index]
                    roll = known_rollnumber[best_index]
                    color = (0, 255, 0)

                    mark_attendance(name, roll, subject)

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            c.rectangle(frame, (left, top), (right, bottom), color, 2)
            c.putText(frame, name, (left, top - 10), c.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        c.imshow("Face Recognition", frame)

        if c.waitKey(1) & 0xFF == 27:
            break

    video.release()
    c.destroyAllWindows()

# ---------------- LOGIN ----------------
def login():
    roll = entry_user.get()
    password = entry_pass.get()

    if roll in student_data and student_data[roll]["password"] == password:
        messagebox.showinfo("Login", f"Welcome {student_data[roll]['name']} ✅")
    else:
        messagebox.showerror("Login", "Invalid Roll or Password ❌")

# ---------------- GUI ----------------
root = tk.Tk()
root.title("Face Attendance System")
root.geometry("400x350")

tk.Label(root, text="Login", font=("Arial", 16)).pack(pady=10)

tk.Label(root, text="Roll Number").pack()
entry_user = tk.Entry(root)
entry_user.pack()

tk.Label(root, text="Password").pack()
entry_pass = tk.Entry(root, show="*")
entry_pass.pack()

tk.Button(root, text="Login", command=login).pack(pady=10)

tk.Label(root, text="Enter Subject Name").pack()
entry_subject = tk.Entry(root)
entry_subject.pack()

tk.Button(root, text="Start Attendance", command=start_recognition).pack(pady=20)

tk.Button(root, text="Exit", command=root.quit).pack()

root.mainloop()
