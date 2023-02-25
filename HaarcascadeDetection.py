import cv2
import numpy as np

def res(final_wide,n,img):
    r = float(final_wide) / img.shape[1]
    dim = (final_wide, int(img.shape[0] * r))
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow(n, resized) 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create the face detecting function 
def detect_face(img):
    img_2 = img.copy()
    face_rects = face_cascade.detectMultiScale(img.copy(), 
                                               scaleFactor = 1.1,
                                               minNeighbors = 4)
    
    for (x, y, w, h) in face_rects:
        cv2.rectangle(img_2, (x, y), (x+w, y+h), (255, 255, 255), 3)
        
    return img_2# Detect the face

# Load the image file and convert the color mode
avengers = cv2.imread('1.png')
avengers = cv2.cvtColor(avengers, cv2.COLOR_BGR2GRAY)# Detect the face and plot the result
detected_avengers = detect_face(avengers)

res(800,"img", detected_avengers)

cv2.waitKey(0)
cv2.destroyAllWindows()
