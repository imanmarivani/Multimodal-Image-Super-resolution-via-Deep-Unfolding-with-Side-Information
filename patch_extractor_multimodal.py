import numpy as np
import cv2
import os, os.path
import glob
import math
from os.path import basename

#image path and valid extensions
imageDir = "./train_images" #specify your path here
image_path_list = []
valid_image_extensions = [".png", ".bmp", ".jpg", ".tiff"] #specify your valid extensions here
valid_image_extensions = [item.lower() for item in valid_image_extensions]
#p1 = 70 #number of vertical patches
#p2 = 50 #number of horizontal patches end up in p1*p2 patches for each w1*w2 image 
n, m = 44, 44 # for a patch size of n*m
 
#create a list all files in directory and
#append files with a vaild extention to image_path_list
for file in os.listdir(imageDir):
    extension = os.path.splitext(file)[1]
    if extension.lower() not in valid_image_extensions:
        continue
    image_path_list.append(os.path.join(imageDir, file))
 
#loop through image_path_list to open each image
cntnir = 0
cntrgb = 0
for imagePath in image_path_list:
    image = cv2.imread(imagePath)
    

    # display the image on screen with imshow()
    # after checking that it loaded
    if image is not None:
        print(image.shape)
        w1 = image.shape[0]
        p1 = math.floor(w1/n)
        w2 = image.shape[1]
        p2 = math.floor(w2/n)   
        #s1 = int(math.floor((w1 - n + 1)/p1))
        s1 = int(math.floor((w1 - n)/max(1,(p1-1))))
        #s2 = int(math.floor((w2 - n + 1)/p2))
        s2 = int(math.floor((w2 - n)/max(1,(p2-1))))
        print (p1, p2)
        
        if imagePath[-7:-4]=="mss":
            for i in range(p1):   # to sweep vertically for patches
                for j in range(p2):  # to sweep horizontally for patches

                    croped_image = image[i*s1:i*s1+n , j*s2:j*s2+m, :]    # ij-th patch
                    for k in [0, 90]:
                        rows, cols, channels = croped_image.shape
                        M = cv2.getRotationMatrix2D((cols/2,rows/2),k,1)
                        dst = cv2.warpAffine(croped_image,M,(cols,rows))
                        #print (dst.shape)
                        name_file =  str(str(cntnir)+"_mss"+".png")
                        cv2.imwrite(name_file, dst)
                        cntnir += 1
                #cv2.imwrite(str(imagePath)+str(i)+str(j)+'.jpg', croped_image)
        if imagePath[-7:-4]=="RGB":
            for i in range(p1):   # to sweep vertically for patches
                for j in range(p2):  # to sweep horizontally for patches

                    croped_image = image[i*s1:i*s1+n , j*s2:j*s2+m, :]    # ij-th patch
                    for k in [0, 90]:
                        rows, cols, channels = croped_image.shape
                        M = cv2.getRotationMatrix2D((cols/2,rows/2),k,1)
                        dst = cv2.warpAffine(croped_image,M,(cols,rows))
                        #print (dst.shape)
                        name_file =  str(str(cntrgb)+"_rgb"+".png")
                        cv2.imwrite(name_file, dst)
                        cntrgb += 1
    #print(cnt)
    
    elif image is None:
        print ("Error loading: " + imagePath)
        #end this loop iteration and move on to next image
        continue
        
    print(cntnir)
    print(cntrgb)