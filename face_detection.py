import cv2 as  c
face_cascade = c.CascadeClassifier(c.data.haarcascades + "haarcascade_frontalface_default.xml")
cap=c.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        print("not read")
        break

    gray = c.cvtColor(frame, c.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        c.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

    c.imshow("face detection", frame)

    if c.waitKey(1) == 27:
        break

cap.release()
c.destroyAllWindows()