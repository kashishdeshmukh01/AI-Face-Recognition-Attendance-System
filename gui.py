from tkinter import *
import subprocess
import sys
from tkinter import *
import os


root = Tk()
root.title("AI Face Recognition Attendance System")
root.geometry("400x400")

Label(
    root,
    text="AI Face Recognition Attendance System",
    font=("Arial", 14, "bold")
).pack(pady=20)

Button(
    root,
    text="Capture Faces",
    width=20,
    command=lambda: subprocess.run([sys.executable, "capture_faces.py"])
).pack(pady=10)

Button(
    root,
    text="Train Model",
    width=20,
    command=lambda: os.system("python train.py")
).pack(pady=10)

Button(
    root,
    text="Start Attendance",
    width=20,
    command=lambda: os.system("python recognizer.py")
).pack(pady=10)

root.mainloop()