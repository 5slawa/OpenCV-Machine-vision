import cv2
import random
def res(final_wide,n,img):
 r = float(final_wide) / img.shape[1]
 dim = (final_wide, int(img.shape[0] * r))
 resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 cv2.imshow(n, resized)
img = cv2.imread("1.jpg")
res(500,"orig",img)
height, width, channels = img.shape
def crop(k,img):
 img2 = img
 img3=[]
 CROP_H_SIZE = int(k**0.5)
 CROP_W_SIZE = int(k/CROP_H_SIZE)
 i=list(range(0, CROP_W_SIZE*CROP_H_SIZE))
 random.shuffle(i)
 n=0
 for ih in range(CROP_H_SIZE ):
 for iw in range(CROP_W_SIZE ):
 x = width/CROP_W_SIZE * iw
 y = height/CROP_H_SIZE * ih
 h = int((height / CROP_H_SIZE))
 w = int((width / CROP_W_SIZE ))
 img = img[int(y):int(y+h), int(x):int(x+w)]
 img3.append(img)
 img = img2

 img = cv2.imread("1.jpg")
 img2 = img

 for ih in range(CROP_H_SIZE):
 for iw in range(CROP_W_SIZE):
 x = width/CROP_W_SIZE * iw
 y = height/CROP_H_SIZE * ih
 h = int((height / CROP_H_SIZE))
 w = int((width / CROP_W_SIZE ))
 img5 = img3[i[n]]
 n+=1
 img2[int(y):int(y+h), int(x):int(x+w)] = img5
 return img2
res(500,"random",crop(24,img))
cv2.waitKey(0)
cv2.destroyAllWindows()
