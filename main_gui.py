# main_gui.py
import tkinter as tk
from tkinter import messagebox
import cv2
import os
import face_recognition
import pickle
import pandas as pd
from datetime import datetime

# ========== FUNCTION 1: Capture Faces ==========
def capture_faces():
    name = entry_name.get()
    if not name:
        messagebox.showerror("Error", "Enter a name before capturing.")
        return

    os.makedirs(f"dataset/{name}", exist_ok=True)
    cap = cv2.VideoCapture(0)
    count = 0
    while count < 10:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Capturing Face Images - Press Q to Quit", frame)
            cv2.imwrite(f"dataset/{name}/{count}.jpg", frame)
            count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Success", f"Captured 10 images for {name}.")

# ========== FUNCTION 2: Train Model ==========
def train_encodings():
    known_encodings = []
    known_names = []

    for name in os.listdir("dataset"):
        for file in os.listdir(f"dataset/{name}"):
            image_path = f"dataset/{name}/{file}"
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(name)

    data = {"encodings": known_encodings, "names": known_names}
    with open("encodings.pkl", "wb") as f:
        pickle.dump(data, f)

    messagebox.showinfo("Training Done", "Face encodings saved successfully.")

# ========== FUNCTION 3: Start Attendance ==========
def start_attendance():
    if not os.path.exists("encodings.pkl"):
        messagebox.showerror("Error", "Train the model first!")
        return

    with open("encodings.pkl", "rb") as f:
        data = pickle.load(f)

    cap = cv2.VideoCapture(0)

    if not os.path.exists("attendance.csv"):
        pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv("attendance.csv", index=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding, box in zip(encodings, boxes):
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"
            if True in matches:
                matched_idx = matches.index(True)
                name = data["names"][matched_idx]

                now = datetime.now()
                date, time = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
                df = pd.read_csv("attendance.csv")
                if not ((df['Name'] == name) & (df['Date'] == date)).any():
                    df.loc[len(df)] = [name, date, time]
                    df.to_csv("attendance.csv", index=False)

            top, right, bottom, left = box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Face Attendance System - Press Q to Quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ========== FUNCTION 4: View Attendance File ==========
def view_attendance():
    if not os.path.exists("attendance.csv"):
        messagebox.showinfo("No Data", "Attendance file not found.")
        return
    os.system("attendance.csv" if os.name == "nt" else "open attendance.csv")

# ========== MAIN GUI ==========
root = tk.Tk()
root.title("Face Recognition Attendance System")
root.geometry("450x350")
root.resizable(False, False)

tk.Label(root, text="Face Recognition Attendance System", font=("Helvetica", 16, "bold")).pack(pady=15)

tk.Label(root, text="Enter Name:", font=("Helvetica", 12)).pack()
entry_name = tk.Entry(root, width=30, font=("Helvetica", 12))
entry_name.pack(pady=5)

tk.Button(root, text="1. Capture Faces", command=capture_faces, width=25, bg="#4CAF50", fg="white").pack(pady=5)
tk.Button(root, text="2. Train Model", command=train_encodings, width=25, bg="#2196F3", fg="white").pack(pady=5)
tk.Button(root, text="3. Start Attendance", command=start_attendance, width=25, bg="#f44336", fg="white").pack(pady=5)
tk.Button(root, text="4. View Attendance File", command=view_attendance, width=25, bg="#FF9800", fg="white").pack(pady=5)

tk.Label(root, text="Press Q to exit camera view", font=("Helvetica", 10, "italic")).pack(pady=10)

root.mainloop()
