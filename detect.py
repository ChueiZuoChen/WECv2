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

# Start OpenCV video streaming.
cap = cv2.VideoCapture(0)
# Check the camera is opened.
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Read stream frames and isolate images and rectangles.
    rectangle, image = cap.read()
    if not rectangle:
        print("Cannot receive frame")
        break
    # Image resize.
    image = cv2.resize(image, (540, 300))
    # Turn images from RGB to Gray level.
    input_gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Fetch face points
    faces = face_cascade.detectMultiScale(input_gray_image)
    # Initial ID enum.
    name = {
        '1': 'Potti',
        '2': 'Didi',
        '3': 'Eric',
    }
    # Looping faces also put labels and draw rectangle on each faces.
    for (x, y, w, h) in faces:
        # draw rectangle on each frame's image.
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Use our model that just trained as the recognizer to predict each face.
        id_number, confidence_level = recognizer.predict(input_gray_image[y:y + h, x:x + w])
        # After predict we will also get the confidence, this one is to identify likelihood for each face.
        if confidence_level < 63:
            # If confidence level reach our target, then we will put the name by id enum on the view.
            print(name[str(id_number)])
            text = name[str(id_number)]
        else:
            text = ''
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('default', image)
    if cv2.waitKey(5) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
