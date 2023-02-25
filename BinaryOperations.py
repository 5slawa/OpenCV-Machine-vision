import cv2
import numpy as np

image = cv2.imread("3.jpg",1)
logo = cv2.imread("2.png",1)

# Я хочу поместить логотип в верхний левый угол, поэтому я создаю ROI
rows,cols,channels = logo.shape
roi = image[0:rows, 0:cols ]

# Теперь создайте маску логотипа и создайте также его инверсную маску
img2gray = cv2.cvtColor(logo,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)

mask_inv = cv2.bitwise_not(mask)

# Теперь затемните область логотипа в ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

# Возьмите только область логотипа из изображения логотипа.
img2_fg = cv2.bitwise_and(logo,logo,mask = mask)

# Поместите логотип в ROI и измените основное изображение
dst = cv2.add(img1_bg,img2_fg)
image[0:rows, 0:cols ] = dst
def res(final_wide,n,img): 
    r = float(final_wide) / img.shape[1]
    dim = (final_wide, int(img.shape[0] * r)) 
    # уменьшаем изображение до подготовленных размеров
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)    
    cv2.imshow(n, resized)
res(500,'image',image)    
res(200,'mask_inv',mask_inv)
res(200,'img1_bg',img1_bg)
res(200,'img2_fg',img2_fg)
res(200,'dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
