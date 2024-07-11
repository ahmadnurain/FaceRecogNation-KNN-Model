import os
import pickle
import numpy as np
import face_recognition
from sklearn.neighbors import KNeighborsClassifier

# Path dataset
dataset_path = 'dataset'

def getEncodingsAndLabels(dataset_path):
    encodings = []
    labels = []
    id_to_name = {}
    current_id = 0

    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            id_to_name[current_id] = folder_name
            for file_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, file_name)
                image = face_recognition.load_image_file(image_path)
                face_encoding = face_recognition.face_encodings(image)

                if len(face_encoding) > 0:
                    encodings.append(face_encoding[0])
                    labels.append(current_id)

            current_id += 1

    return np.array(encodings), np.array(labels), id_to_name

# Mendapatkan encodings dan label dari semua folder dalam dataset
encodings, labels, id_to_name = getEncodingsAndLabels(dataset_path)

# Membuat dan melatih model KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(encodings, labels)

# Menyimpan model yang telah dilatih
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Menyimpan peta ID ke nama
with open('id_to_name.pkl', 'wb') as f:
    pickle.dump(id_to_name, f)

print("Training completed and model saved.")
