import numpy as np
import cv2
import face_recognition
import os
from datetime import datetime
path= 'images'
images=[]
classNames=[]
list1=os.listdir(path)
print(list1)
for clas in list1:
    myImage= cv2.imread(f'{path}/{clas}')
    images.append(myImage)
    classNames.append(os.path.splitext(clas)[0])
print(classNames)

def findEncodings(images):
    encodedlist=[]
    for img in images:
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128), None, 0.25, 0.25)
        encode_image = face_recognition.face_encodings(img)[0]
        encodedlist.append(encode_image)
    return encodedlist
encodedlist_1=findEncodings(images)
print(len(encodedlist_1))

def mark_attendance(name):
    with open('class_attendance.csv','r+') as f:
        my_list1=f.readlines()
        list_2=[]
        for line in my_list1:
            line_1= line.split(',')
            list_2.append(line_1[0])
        if name not in list_2:
            now=datetime.now()
            exact_time=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{exact_time}')



img_1= face_recognition.load_image_file('docs/photo_2022-11-11_16-47-47.jpg')
img_1= cv2.cvtColor(img_1,cv2.COLOR_BGR2RGB)
img_1=cv2.resize(img_1,(400,300))

faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = faceCascade.detectMultiScale(img_1,1.1,4)
encode_liveimgs= face_recognition.face_encodings(img_1, faces)

for encodefaces, faceLoc in zip(encode_liveimgs, faces):
    matches = face_recognition.compare_faces(encodedlist_1, encodefaces, tolerance=0.7)
    faceDis = face_recognition.face_distance(encodedlist_1, encodefaces)
    print(faceDis)
    matchIndex = np.argmin(faceDis)
    name = "unknown"
    if matches[matchIndex]:
        name = classNames[matchIndex].upper()
        print(name)
for(x,y,w,h) in faces:
    cv2.rectangle(img_1,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.putText(img_1, name, (x+w, y+h), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
    mark_attendance(name)
cv2.imshow('my image',img_1)
cv2.waitKey(0)



# cap=cv2.VideoCapture(0)
# cap.set(3,400)
# cap.set(4,400)
# while True:
#     success, img= cap.read()
#     imgs=cv2.resize(img,(400,400),None,0.25)
#     imgs=cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
#     faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     faces = faceCascade.detectMultiScale(imgs, 1.1, 4)
#     for (x, y, w, h) in faces:
#         cv2.rectangle(imgs, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(imgs, 'name', (x+w, y+h), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
#     cv2.imshow('live', imgs)
#     if cv2.waitKey(1) & 0xFF ==ord('s'):
#         break
