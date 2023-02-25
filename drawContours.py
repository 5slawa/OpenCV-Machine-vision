import numpy as np
import cv2
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

img = cv2.imread("scale_1200.jpg")
res(300,"img", img)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_blue = np.array([0,0,24])
upper_blue = np.array([200,128,250])

mask = cv2.inRange(hsv, lower_blue, upper_blue)

nmask = cv2.bitwise_not(mask)
fon=cv2.bitwise_and(mask, nmask)

kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(nmask, cv2.MORPH_OPEN, kernel)

contours, hier = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img1=cv2.drawContours(fon,contours, -1, (255,0,0), 2, cv2.LINE_AA, hier, 1 ) 
cv2.drawContours(img,contours, -1, (255,255,0), 5, cv2.LINE_AA, hier, 1 ) 

res(300,"mask", mask)
res(300,"nmask", nmask)
res(300,"img1", img1)

cv2.waitKey(0)
cv2.destroyAllWindows()
