import cv2
import numpy as np
import face_recognition
img_1= face_recognition.load_image_file('images/ugyen.jpg')
img_1= cv2.cvtColor(img_1,cv2.COLOR_BGR2RGB)
img_1=cv2.resize(img_1,(400,500))
face_enccode= face_recognition.face_encodings(img_1)[0]

faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = faceCascade.detectMultiScale(img_1,1.1,4)
for(x,y,w,h) in faces:
    cv2.rectangle(img_1,(x,y),(x+w,y+h),(0,255,0),2)
#cv2.imshow('my image',img_1)
#cv2.waitKey(0)
img_11= face_recognition.load_image_file('images/kalab.jpg')
img_11= cv2.cvtColor(img_11,cv2.COLOR_BGR2RGB)
img_11=cv2.resize(img_11,(400,500))
faces2 = faceCascade.detectMultiScale(img_11,1.1,4)
for(x,y,w,h) in faces2:
    cv2.rectangle(img_11,(x,y),(x+w,y+h),(0,255,0),2)
#cv2.imshow('my image2',img_11)
#cv2.waitKey(0)
encode_test=face_recognition.face_encodings(img_11)[0]
results=face_recognition.compare_faces([encode_test],face_enccode,tolerance=0.4)
face_dis=face_recognition.face_distance([encode_test],face_enccode)
print(results,face_dis)
cv2.putText(img_11,f'{results} {round(face_dis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
cv2.imshow('kal',img_11)
cv2.waitKey(0)



