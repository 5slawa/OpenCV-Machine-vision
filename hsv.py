import cv2
import math
import numpy as np
def res(i,n,img):
    i=math.pi*i
    h, w = img.shape[:2]   
    dim=(int(i*w), int(i*h))
    resized = cv2.resize(img,dim, interpolation = cv2.INTER_AREA)
    cv2.imshow(n, resized)    
    
def rot(angle, scale, img):
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

def mos(img):
     image = cv2.imread(img)
     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
     h = hsv[:,:,0]
     s = hsv[:,:,1]
     v = hsv[:,:,2]
     image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     hei, wid = image.shape[:2]
     hei=int(hei)
     wid=int(wid)
     h2=int(hei/2)
     w2=int(wid/2)
     img2 = h[0:h2, 0:w2]
     image[0:h2, 0:w2] = img2
     img3 = s[h2:hei, w2:wid]
     image[h2:hei, w2:wid] = img3
     img4 = v[h2:hei, 0:w2]
     image[h2:hei, 0:w2] = img4
     img5 = image[0:h2, w2:wid]
     image[0:h2, w2:wid] = img5
     res(.1,"h", img2)
     res(.1,"s", img3)
     res(.1,"v", img4)
     res(.1,"orig", img5)
     img=rot(-30,1,image)
     res(.1,"rot", img)
     
def mos2(img):
    image = cv2.imread(img)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hei, wid = image.shape[:2] 
    hei=int(hei)
    wid=int(wid)    
    img = np.zeros((2*hei,2*wid,3), np.uint8)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    imgh = h[0:hei, 0:wid]
    img[0:hei, 0:wid] = imgh
    
    imgs = s[0:hei, 0:wid]
    img[hei:2*hei, wid:2*wid] = imgs
    
    imgv = v[0:hei, 0:wid]
    img[hei:2*hei, 0:wid] = imgv
    
    imgo = image[0:hei, 0:wid]
    img[0:hei, wid:2*wid] = imgo
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,'v',(1500,wid), font,10,(500,255,0),6,cv2.LINE_AA)
    cv2.putText(img,'orig',(3*hei,300), font,10,(0,0,0),6,cv2.LINE_AA)
    cv2.putText(img,'h',(1500,300), font,10,(500,255,0),6,cv2.LINE_AA)
    cv2.putText(img,'s',(3*hei,wid), font,10,(500,255,0),6,cv2.LINE_AA)
    
    res(.05,"h", imgh)
    res(.05,"s", imgs)
    res(.05,"v", imgv)
    res(.05,"orig", imgo)
    img=rot(-30,0.7,img)
    res(.1,"rot", img) 
   
    
mos2("1.jpg")  

cv2.waitKey(5000) 
cv2.destroyAllWindows()
