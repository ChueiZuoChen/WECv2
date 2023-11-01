# 10571545 Huajian Liang
# 10511047 Chuei-Zuo Chen
# 10579916 Di Wu

# Load openCV library
import cv2

# Use OpenCV FaceRecognizer.
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Load the model that we trained before.
recognizer.read('face.yml')
# Load face detector model, to fetch all the faces points and head rectangle.
cascade_path = "xml/haarcascade_frontalface_default.xml"
# Use face detector model to mapping the OpenCV video streaming frame to recognise which face is belone to which person.
face_cascade = cv2.CascadeClassifier(cascade_path)

for i in range(1,31):
    image = cv2.imread(f'face_test/{i}.jpg')
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    name = {
        '1': 'Potti',
        '2': 'Didi',
        '3': 'Eric',
    }
    sttr = "image" + str(i)
    print(sttr)
    # Looping faces also put labels and draw rectangle on each face.
    for (x, y, w, h) in faces:
        id_number, confidence_level = recognizer.predict(gray[y:y + h, x:x + w])
        if confidence_level < 75:
            print(name[str(id_number)])
