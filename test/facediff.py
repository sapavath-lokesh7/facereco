# importing librarys
from base64 import encode
from multiprocessing.connection import wait
import cv2
import numpy as n
import face_recognition as fr

#resziging image

def res(img, size):
    width = int(img.shape[1]*size)
    height = int(img.shape[0]*size)
    dimension = (width,height)
    return cv2.resize(img,dimension, interpolation= cv2.INTER_AREA)

#real img
lokesh = fr.load_image_file("imgs/loki.jpg")
lokesh =cv2.cvtColor(lokesh, cv2.COLOR_BGR2RGB)
lokesh =res(lokesh,0.40)
#test img
lokesh_test = fr.load_image_file("imgs/loki_test.jpg")
lokesh_test =cv2.cvtColor(lokesh_test, cv2.COLOR_BGR2RGB)
lokesh_test =res(lokesh_test,0.40)

#finding face location
facelocation_lokesh = fr.face_locations(lokesh)[0]
encode_lokesh = fr.face_encodings(lokesh)[0]
cv2.rectangle(lokesh,(facelocation_lokesh[3],facelocation_lokesh[0]),(facelocation_lokesh[1],facelocation_lokesh[2]),(255,255,0),2)
facelocation_lokesh_test = fr.face_locations(lokesh_test)[0]
encode_lokesh_test = fr.face_encodings(lokesh_test)[0]
cv2.rectangle(lokesh_test,(facelocation_lokesh_test[3],facelocation_lokesh_test[0]),(facelocation_lokesh_test[1],facelocation_lokesh_test[2]),(255,255,0),2)

comp = fr.compare_faces([encode_lokesh],encode_lokesh_test)
cv2.putText(lokesh_test, f'{comp}', (50,50),cv2.FONT_ITALIC,1,(0,0,255),2)

# display tested image
cv2.imshow('main_img',lokesh)
cv2.imshow('test_img',lokesh_test)

cv2.waitKey(0)
cv2.destroyAllWindows()