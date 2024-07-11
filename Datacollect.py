import cv2
import os

facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(0)

id = input("id: ")
name = input("name: ")

# Membuat folder dengan nama sesuai inputan jika belum ada
folder_path = 'dataset/' + name
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

count = 0

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        count += 1
        
        # Menyimpan gambar di folder sesuai dengan inputan name
        cv2.imwrite(os.path.join(folder_path, str(name) + '.' + str(id) + '.' + str(count) + ".jpg"), gray[y:y+h, x:x+w])
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
        
    cv2.imshow("Frame", frame)
    
    k = cv2.waitKey(1)
    
    if count > 500:
        break

video.release()
cv2.destroyAllWindows()
print("Data collection completed.")
