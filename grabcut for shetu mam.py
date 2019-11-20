import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt


mask=np.zeros([224,224] , dtype=np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
rect= (25,25,150,175)
kernal = np.ones((10, 10), np.uint8)
kernel1 = np.ones((5, 5), np.uint8)


# code of grabcuting
img=cv2.resize(cv2.imread('/home/ashrafi/Documents/ISIC_0028688.jpg'), (224,224))
close=cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernal) #here operation for hear removeing

cv2.imshow('here remove',close)
clone= close.copy()

cv2.grabCut(close, mask,rect,bgdModel,fgdModel,1, cv2.GC_BGD)
mask2=np.where((mask==2)|(mask==0),0,1).astype('uint8')
out= close*mask2[:, :, np.newaxis]
# code for channel changeing
cv2.imshow('grabcut', out)

gray=cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
_, ther= cv2.threshold(gray, 1, 255 ,cv2.THRESH_BINARY)
cv2.imshow('ther', ther)

close1= cv2.morphologyEx(ther, cv2.MORPH_CLOSE, kernel1)  #here operation for remove image nois
cv2.imshow('nois remove', close1)
#code for find contros
contours= cv2.findContours(close1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for contour in contours[0]:
    x, y, w, h= cv2.boundingRect(contour)
    rec= cv2.rectangle(clone, (x,y), (x+w, y+h), (0,255,0), 4)
    cv2.imshow('rectangle', rec)
    crop_img=clone[y:y+h, x:x+w]
    cv2.imshow('crop img', crop_img)
cv2.waitKey(0)
cv2.destroyAllWindows()