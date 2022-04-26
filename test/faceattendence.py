import cv2
import numpy as n
import face_recognition as fr
import os
import pyttsx3 as ts
from datetime import  datetime

#to convert text to speach
engine = ts.init()

#to resize images 
def resize(img, size) :
    width = int(img.shape[1]*size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)

#defining paths and creating list to store pics and names
path = 'student_images'
studentimg = []
studentname = []
mylist = os.listdir(path)

#adding pics and names to list and splitting the extensions
for stud in mylist :
    curimg = cv2.imread(f'{path}/{stud}')
    studentimg.append(curimg)
    studentname.append(os.path.splitext(stud)[0])

#encoding images and converting images bgr to rgb
def findencoding(images) :
    imgencodings = []
    for img in images :
        img = resize(img, 0.50)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeimg = fr.face_encodings(img)[0]
        imgencodings.append(encodeimg)
    return imgencodings

#recording attendence to csv file or msxl
def recordattendence(name):
    with open('attendence.csv', 'r+') as f:
        myDatalist =  f.readlines()
        nameList = []
        for line in myDatalist :
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            timestr = now.strftime('%H:%M, %b:%d')
            f.writelines(f'\n{name}, {timestr}')
            statment = str('welcome to class' + name)
            engine.say(statment)
            engine.runAndWait()




EncodeList = findencoding(studentimg)

# starting video
vid = cv2.VideoCapture(0)

#verifiying face in the frame and recognization
while True :
    success, frame = vid.read()
    Smaller_frames = cv2.resize(frame, (0,0), None, 0.25, 0.25)

    facesinframe = fr.face_locations(Smaller_frames)
    encodefacesinframe = fr.face_encodings(Smaller_frames, facesinframe)

    for encodeFace, faceloc in zip(encodefacesinframe, facesinframe) :
        matches = fr.compare_faces(EncodeList, encodeFace)
        facedis = fr.face_distance(EncodeList, encodeFace)
        print(facedis)
        matchindex = n.argmin(facedis)

        if matches[matchindex] :
            name = studentname[matchindex].upper()
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2-25), (x2, y2), (0, 255, 255), cv2.FILLED)
            cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_ITALIC, 0.7, (0, 0, 255), 1)
            recordattendence(name)

    cv2.imshow('video',frame)
    cv2.waitKey(1)