import numpy as np
import cv2
import os
import glob

def image_opening(gray):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
    res = cv2.morphologyEx(gray,cv2.MORPH_OPEN,kernel)
    return res
    
def crop(img, crop_length):
    img_height, img_width = img.shape
    crop_img = img[crop_length:img_height-crop_length, crop_length:img_width-crop_length]
    crop_img = cv2.resize(crop_img, (img_height, img_width), interpolation=cv2.INTER_CUBIC)
    return crop_img

def createLargest(img):
    new_img = np.zeros_like(img)                                        
    for val in np.unique(img)[1:]:                                      
        mask = np.uint8(img == val)                                     
        labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]  
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])      
        new_img[labels == largest_label] = val                          
    return new_img

if __name__ == "__main__":
    src = glob.glob('*.png')
    print(src)
    for f in src:
        print(f)
        img = cv2.imread(f, 0)
        img = image_opening(img)
        img = crop(img, 30)
        print("img:", img.shape)
        thresh = (img > 0.7*255.0)
        print(thresh.shape)
        thresh = cv2.resize(np.float32(thresh), thresh.shape)
        img = createLargest(thresh)
        cv2.imshow('result', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('post_'+f, 255.0*img)
