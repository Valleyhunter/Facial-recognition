import cv2
import os
import numpy as np
import json

# Base directory for the project
base_dir = r'C:\Users\user\Desktop\studies\AI PROJ\opencv_cam_project'

# Directory for storing captured face images
image_dir = os.path.join(base_dir, 'images')

# File for saving label-to-name mapping
label_info_file = os.path.join(base_dir, 'label_info.json')

# File to save the trained model
model_path = os.path.join(base_dir, 'face_recognizer_model.yml')

# Create necessary directories and files
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

if not os.path.exists(label_info_file):
    with open(label_info_file, 'w') as f:
        json.dump({}, f)

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load or initialize label-to-name mapping
with open(label_info_file, 'r') as f:
    label_info_map = json.load(f)

# Function to detect a face in an image
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if len(faces) == 0:
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y+h, x:x+w], (x, y, w, h)

# Function to prepare training data
def prepare_training_data(folder):
    faces, labels = [], []
    for filename in os.listdir(folder):
        if not filename.endswith(".jpg"):
            continue
        label = int(filename.split('.')[0])  # Extract label from filename
        img_path = os.path.join(folder, filename)
        image = cv2.imread(img_path)
        face, rect = detect_face(image)
        if face is not None:
            faces.append(face)
            labels.append(label)
    return faces, np.array(labels)

# Enroll new person
name = input("Enter your name: ")
roll_number = input("Enter your roll number: ")
label = len(label_info_map) + 1  # Assign next label
label_info_map[str(label)] = f"{name} ({roll_number})"  # Ensure label is string for JSON compatibility

# Save updated label-info map
with open(label_info_file, 'w') as f:
    json.dump(label_info_map, f)

print("Look at the camera and press 'q' to capture images for training.")
capture_count = 0

while capture_count < 60:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    for (x, y, w, h) in faces_detected:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow("Video", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        if len(faces_detected) > 0:
            (x, y, w, h) = faces_detected[0]
            cropped_face = gray[y:y+h, x:x+w]
            face_filename = os.path.join(image_dir, f"{label}.{capture_count}.jpg")
            cv2.imwrite(face_filename, cropped_face)
            capture_count += 1
            print(f"Captured {capture_count} images.")
        else:
            print("No face detected. Try again.")

video_capture.release()
cv2.destroyAllWindows()

# Train face recognizer
print("Preparing training data...")
faces, labels = prepare_training_data(image_dir)

# Create the LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, labels)

# Save trained model
face_recognizer.save(model_path)
print("Training complete and model saved.")
