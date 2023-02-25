import numpy as np
import cv2
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 20})
plt.figure().set_size_inches(20, 15)
def res(final_wide,n,img):
    r = float(final_wide) / img.shape[1]
    dim = (final_wide, int(img.shape[0] * r))
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow(n, resized)
    
def rot(angle, scale, img):
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated



#roi is the object or region of object we need to find
roi = cv2.imread('2.jpg')
hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
target = cv2.imread('1.jpg')
hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
M = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
I = cv2.calcHist([hsvt],[0, 1], None, [180, 256], [0, 180, 0, 256] )
R=M/I
h,s,v = cv2.split(hsvt)
B = R[h.ravel(),s.ravel()]
B = np.minimum(B,1)
B = B.reshape(hsvt.shape[:2])
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
cv2.filter2D(B,-1,disc,B)
B = np.uint8(B)
cv2.normalize(B,B,0,255,cv2.NORM_MINMAX)
ret,thresh = cv2.threshold(B,50,255,0)
thresh = cv2.merge((thresh,thresh,thresh))
res1 = cv2.bitwise_and(target,thresh)
res1 = np.vstack((target,thresh,res1))
cv2.imwrite('res.jpg',res1)

#res(300,"hist", hist)

#cv2.waitKey(0)
#cv2.destroyAllWindows()
