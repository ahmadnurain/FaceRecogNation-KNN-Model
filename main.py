import cv2
import numpy as np
import os
import pickle
import face_recognition
from sklearn.neighbors import KNeighborsClassifier

# Load trained KNN model and ID-to-name mapping
with open('knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)
with open('id_to_name.pkl', 'rb') as f:
    id_to_name = pickle.load(f)

# Load pre-trained age and gender models
age_net = cv2.dnn.readNetFromCaffe(
    'age_deploy.prototxt', 
    'age_net.caffemodel'
)
gender_net = cv2.dnn.readNetFromCaffe(
    'gender_deploy.prototxt', 
    'gender_net.caffemodel'
)

# Define lists for age and gender
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Initialize video capture
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Threshold for unknown faces
distance_threshold = 0.6

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w].copy()
        
        # Encode the face
        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_face)
        
        if len(encodings) > 0:
            encoding = encodings[0]
            encoding_reshaped = encoding.reshape(1, -1)
            distances, indices = knn_model.kneighbors(encoding_reshaped, n_neighbors=1)
            closest_distance = distances[0][0]
            name_pred = knn_model.predict(encoding_reshaped)[0]
            name = id_to_name.get(name_pred, "Unknown")
            
            if closest_distance > distance_threshold:
                name = "Unknown"
            
            # Prepare input blob for age and gender detection
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

            # Predict gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]

            # Predict age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]

            label = f"{name}, {gender}, {age}"
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
print("Face recognition with age and gender prediction completed.")
