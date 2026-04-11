import cv2 as c
import face_recognition as fr
import pandas as pd
import datetime as dt

# Ask for subject name (this will be used as the CSV filename)
subject = input("Enter subject name (CSV filename): ")

# Known face encodings and metadata
known_face_encodings = []
known_rollnumber = []
known_names = []

# Helper function to add known faces safely
def add_known_face(image_path, name, rollnumber):
    image = fr.load_image_file(image_path)
    encodings = fr.face_encodings(image)
    if encodings:
        known_face_encodings.append(encodings[0])
        known_names.append(name)
        known_rollnumber.append(rollnumber)

# Add all faces
add_known_face("abhijeet.jpeg", "Abhijeet", "125110023")
add_known_face("kshitij.jpeg", "Kshitij", "125110012")
add_known_face("yash.jpeg", "Yash", "125110009")
add_known_face("yuvraj.jpeg", "Yuvraj", "125110057")
add_known_face("harshvardhan.jpeg", "Harshvardhan", "125110001")
add_known_face("abhishek.jpeg", "Abhishek", "125110045")
add_known_face("anshu.jpeg", "Anshu", "125110033")   # FIXED: was wrongly using image6
add_known_face("balwant.jpeg", "Balwant", "125110055")
add_known_face("mahima.jpeg", "Mahima", "125110030")
add_known_face("nupur.jpeg", "Nupur", "125110050")
add_known_face("sahil.jpeg", "Sahil", "125110051")
add_known_face("tanvi.jpeg", "Tanvi", "125110028")
add_known_face("ujjwala.jpeg", "Ujjwala", "125110031")
add_known_face("raunak.jpeg", "Raunak", "125110013")
add_known_face("anand.jpeg", "Anand", "125110002")
add_known_face("satyam.jpeg", "Satyam", "125110039")
add_known_face("anjali.jpeg", "Anjali", "125310001")
add_known_face("divyanshu.jpeg", "Divyanshu", "125110014")

# Initialize webcam
video = c.VideoCapture(0)
video.set(c.CAP_PROP_FRAME_HEIGHT, 1000)
video.set(c.CAP_PROP_FRAME_WIDTH, 1000)
video.set(c.CAP_PROP_FPS, 30)

attendance_mark = {}
attendance_df = pd.DataFrame(columns=["name", "date&time", "rollnumber"])

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Resize frame for faster processing
    small_frame = c.resize(frame, (0, 0), fx=0.25, fy=0.25)  # FIXED: better balance
    rgb_small_frame = c.cvtColor(small_frame, c.COLOR_BGR2RGB)

    # Detect faces
    face_locations = fr.face_locations(rgb_small_frame, model="hog")
    face_encodings = fr.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = fr.compare_faces(known_face_encodings, face_encoding)
        face_distances = fr.face_distance(known_face_encodings, face_encoding)
        best_match_index = face_distances.argmin()

        # Scale back up face locations since frame was resized
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        color = (255, 0, 0)
        name = "Unknown"
        roll = None

        if matches[best_match_index]:
            name = known_names[best_match_index]
            roll = known_rollnumber[best_match_index]
            color = (0, 255, 0)

        # Draw rectangle and name
        c.rectangle(frame, (left, top), (right, bottom), color, 2)
        c.putText(frame, name, (left, top - 10), c.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Mark attendance
        if roll is not None and roll not in attendance_mark:
            attendance_mark[roll] = "true"
            attendance_df.loc[len(attendance_df)] = [name, dt.datetime.now(), roll]
            attendance_df.to_csv(subject, mode="a", index=False, header=False)

    # Update class data file
    try:
        df_2 = pd.read_csv("C:/Users/Harshvardhan Singh R/New folder/python/class_data.csv")
        df_2.columns = ["name", "roll no."]

        if "name" in df_2.columns:
            timestamp = str(dt.datetime.now().date())
            df_2[timestamp] = df_2["name"].isin(attendance_df["name"])
            df_2[timestamp] = df_2[timestamp].map({True: "Present", False: "Absent"})
            df_2.to_csv(subject, index=False)  # FIXED: ensure index=False
        else:
            print("No 'name' column found in class_data.csv")
    except Exception as e:
        print("Error reading class_data.csv:", e)

    # Show video
    c.imshow("Fast Face Detection", frame)

    # Exit on ESC key
    if c.waitKey(1) & 0xFF == 27:
        break

video.release()
c.destroyAllWindows()
