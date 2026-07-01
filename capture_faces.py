import cv2
import os

person_name = input("Enter person's name: ")

dataset_path = f"dataset/{person_name}"
os.makedirs(dataset_path, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        count += 1
        file_name = f"{dataset_path}/{count}.jpg"
        cv2.imwrite(file_name, face)

        cv2.rectangle(frame, (x, y), (x+w, y+h),
                      (0, 255, 0), 2)

        cv2.putText(frame, f"Images: {count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

    cv2.imshow("Capture Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 100:
        break

cap.release()
cv2.destroyAllWindows()
print("Dataset collection completed.")