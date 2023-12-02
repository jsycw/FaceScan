import numpy as np
import cv2
import os
import time

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0)

# Folder setup
folder_path = r"C:\Users\jessy\Desktop\side projects\FaceScan\Data"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

face_detected = False

# Request name input for saving the image
name = input("Nama mahasiswa: ").strip()
filename = f"{name}.png"

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        full_path = os.path.join(folder_path, filename)
        cv2.imwrite(full_path, frame)

        face_detected = True
        break

    if face_detected:
        print(f"Saved face image as {full_path}")
        break

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
