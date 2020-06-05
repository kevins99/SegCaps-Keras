from gen_map import gen_map
import os
import cv2

DIR_LIST = ['./data_example/train_imgs/train', './data_example/test_imgs/test', './data_example/val_imgs/val']

dirlist = []
filelist = []
for dir in DIR_LIST:
    filelist = os.listdir(dir)
    filelist = [dir + '/' + f for f in filelist]
    dirlist.append(filelist)

for filelist in dirlist:
    for file in filelist:
        gray_img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        rgb_img = gen_map(gray_img)
        cv2.imwrite(file, rgb_img)
