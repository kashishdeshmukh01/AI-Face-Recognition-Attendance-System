import cv2
import os
import numpy as np

dataset_path = "dataset"

faces = []
labels = []
label_ids = {}
current_id = 0

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_folder):
        continue

    label_ids[current_id] = person_name

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        faces.append(img)
        labels.append(current_id)

    current_id += 1

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
recognizer.save("trainer/trainer.yml")

print("Model trained successfully!")
print(label_ids)