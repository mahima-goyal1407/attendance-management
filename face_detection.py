import cv2 as c
import face_recognition as fr

image = fr.load_image_file("anshu.jpeg")
known_face_encodings = fr.face_encodings(image)

video = c.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret: break

    
    small_frame = c.resize(frame, (0, 0), fx=0.20, fy=0.20)
    
    rgb_small_frame = c.cvtColor(small_frame, c.COLOR_BGR2RGB)

    
    face_locations = fr.face_locations(rgb_small_frame)
    face_encodings = fr.face_encodings(rgb_small_frame, face_locations)# returns a list of encodings

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = fr.compare_faces(known_face_encodings, face_encoding)
        
        
        top *= 5
        right *= 5
        bottom *= 5
        left *= 5

        color = (0, 255, 0) if True in matches else (0, 0, 255)
        name = "Matched" if True in matches else "Unknown"

        c.rectangle(frame, (left, top), (right, bottom), color, 2)
        c.putText(frame, name, (left, top - 10), c.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    c.imshow("Fast Face Detection", frame)

    if c.waitKey(1) & 0xFF == 27:
        break

video.release()
c.destroyAllWindows()