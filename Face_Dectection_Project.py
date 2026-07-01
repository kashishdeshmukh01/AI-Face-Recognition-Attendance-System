import cv2
import csv
from datetime import datetime

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

# Names according to your training output
names = {
    0: "kash",
    1: "ritik"
}

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

marked = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]

        label, confidence = recognizer.predict(face)

        if confidence < 70:
            name = names.get(label, "Unknown")

            cv2.putText(
                frame,
                name,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Mark attendance once
            if name not in marked:
                now = datetime.now()

                with open("attendance.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        name,
                        now.strftime("%Y-%m-%d"),
                        now.strftime("%H:%M:%S")
                    ])

                marked.add(name)

        else:
            cv2.putText(
                frame,
                "Unknown",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (255, 0, 0),
            2
        )

    cv2.imshow("AI Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()