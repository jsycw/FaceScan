import face_recognition
import cv2
import os
import csv
import numpy as np
from datetime import datetime

def load_known_faces(data_folder):
    known_faces = []
    known_names = []

    for filename in os.listdir(data_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(data_folder, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]

            known_faces.append(encoding)
            known_names.append(filename[:-4])

    return known_faces, known_names

def save_attendance(name):
    try:
        with open("attendance.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    except Exception as e:
        print(f"Error writing to attendance file: {e}")


def main():
    data_folder = r"C:\Users\jessy\Desktop\side projects\FaceScan\Data"
    known_faces, known_names = load_known_faces(data_folder)

    cap = cv2.VideoCapture(0)
    stop_on_no_match = True  # Set this to False if you want continuous scanning without stopping

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_matched = False
        for face_encoding in face_encodings:
            if len(face_locations) == 0:
                print("No faces detected in the frame.")
                continue

            distances = face_recognition.face_distance(known_faces, face_encoding)
            if len(distances) > 0 and min(distances) < 0.6:  # Threshold for a match
                best_match_index = np.argmin(distances)
                name = known_names[best_match_index]
                save_attendance(name)
                print(f"Attendance recorded for {name}")
                face_matched = True
                break

        if face_matched:
            break

        if not face_matched and face_encodings:
            print("No match found.")  # Handle no match case
            if stop_on_no_match:
                break

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
