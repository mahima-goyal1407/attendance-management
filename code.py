import cv2 as c
import face_recognition as fr
import pandas as pd
import datetime as dt
subject=input("enter subject name:")
known_face_encodings =[]
known_rollnumber=[]
known_names=[]
image = fr.load_image_file("anshu.jpeg")
encoding1=fr.face_encodings(image)[0]
known_names.append("Anshu")
known_face_encodings.append(encoding1)
known_rollnumber.append("125110033")
image2=fr.load_image_file("shubham.jpeg")
encoding2=fr.face_encodings(image2)[0]
known_names.append("Shubham")
known_face_encodings.append(encoding2)
known_rollnumber.append(125114029)

video = c.VideoCapture(0)
video.set(c.CAP_PROP_FRAME_HEIGHT,240)
video.set(c.CAP_PROP_FRAME_WIDTH,320)
video.set(c.CAP_PROP_FPS,30)
matches=[]
attendance_mark={}

while True:
    ret, frame = video.read()
    if not ret: break

    #to read easily and remove lag
    small_frame = c.resize(frame, (0, 0), fx=0.1, fy=0.1)
    #face recognition work in rgb
    rgb_small_frame = c.cvtColor(small_frame, c.COLOR_BGR2RGB)

    #this gives the face dimensions
    face_locations = fr.face_locations(rgb_small_frame,model="hog")
    face_encodings = fr.face_encodings(rgb_small_frame,face_locations)# returns a list of encodings
    #here we are detecting every frame the webcame capture
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = fr.compare_faces(known_face_encodings, face_encoding)# return a list of boolean values
        
        
        top *= 10
        right *= 10
        bottom *= 10
        left *= 10
        #if else condition
        color = (255, 0, 0)
        name="Unknown"
        roll=None
        if True in matches:
            match_index=matches.index(True)
            name=known_names[match_index]
            roll=known_rollnumber[match_index]
            color=(0,255,0)

        c.rectangle(frame, (left, top), (right, bottom), color, 2)
        c.putText(frame, name, (left, top - 10), c.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        c.imshow("Fast Face Detection", frame)
        if True in matches and roll is not None and roll not in attendance_mark:
            attendance_mark[roll]="true"
            data={"name":name,"date&time":[dt.datetime.now()],"rollnumber":roll}
            df=pd.DataFrame(data)
            df.to_csv(subject,mode="a",index=False,header=False)
    # val=pd.read_csv("attendance.csv")
    # val.drop_duplicates(inplace=True)
    # val.to_csv("attendance.csv",index=False,header=False)
    df_2=pd.read_csv("C:\\Users\\Anshu\\OneDrive\\Documents\\class data.csv",header=None)
    df_2.columns=["name","roll no."]
    if "name" in df_2.columns:
        timestamp = str(dt.datetime.now().date())

        df_2[timestamp] = df_2["name"].isin(df["name"])
        df_2[timestamp] = df_2[timestamp].map({True:"Present", False:"Absent"})

        df_2.to_csv(subject)
    else:
        print("no column")

    if c.waitKey(1) & 0xFF == 27:
        break

video.release()
c.destroyAllWindows()