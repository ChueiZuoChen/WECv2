# 10571545 Huajian Liang
# 10511047 Chuei-Zuo Chen
# 10579916 Di Wu

import cv2
import numpy as np

if __name__ == '__main__':
    # Load face detector model, to fetch all the faces points and head rectangle.
    detector = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')
    # OpenCV facial recognizer
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_detect_result_collection = []
    face_id = []

    # Load face01 images(Potti's image dataset)
    for i in range(1, 29):
        # Read image
        img = cv2.imread(f'face01/{i}.jpg')
        # Turn image from RBG to Gray Scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Save image into unit8 data format (PyNum)
        img_np = np.array(gray, 'uint8')
        # Using open cv CascadeClassifier to detect image, it will return all the face points location and the head rectangle.
        face = detector.detectMultiScale(gray)
        for (x, y, w, h) in face:
            # Collect face data
            face_detect_result_collection.append(img_np[y:y + h, x:x + w])
            # Temp list for store the face ID
            face_id.append(1)

    # Load face02 images(Didi's image dataset)
    for i in range(1, 30):
        # Read image
        img = cv2.imread(f'face02/{i}.jpg')
        # Turn image from RBG to Gray level
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Save image into unit8 data format
        img_np = np.array(gray, 'uint8')
        # Using open cv CascadeClassifier to detect image, it will return all the face points location and the head rectangle.
        face = detector.detectMultiScale(gray)
        for (x, y, w, h) in face:
            # Collect face data
            face_detect_result_collection.append(img_np[y:y + h, x:x + w])
            # Temp list for store the face ID
            face_id.append(2)

    # Load face03 images(Eric's image dataset)
    for i in range(1, 31):
        # Read image
        img = cv2.imread(f'face03/{i}.jpg')
        # Turn image from RBG to Gray level
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Save image into unit8 data format
        img_np = np.array(gray, 'uint8')
        # Using open cv CascadeClassifier to detect image, it will return all the face points location and the head rectangle.
        face = detector.detectMultiScale(gray)
        for (x, y, w, h) in face:
            # Collect face data
            face_detect_result_collection.append(img_np[y:y + h, x:x + w])
            # Temp list for store the face ID
            face_id.append(3)

    print('training model...')
    # Here is the ML training entry point, feed face detection's result collections and face id to it for training ML model.
    face_recognizer.train(face_detect_result_collection, np.array(face_id))
    # Save new model that we just trained.
    face_recognizer.save('face.yml')
    print('ok!')
