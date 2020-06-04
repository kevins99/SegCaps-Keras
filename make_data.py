import os 
import random 
import re 

from PIL import Image

DATA_PATH = 'data'
IMG_PATH = DATA_PATH+'/imgs/'
MASK_PATH = DATA_PATH+'/masks/'

folders = ['train_imgs', 'train_masks', 'val_imgs', 'val_masks', 'test_imgs', 'test_masks']

for folder in folders:
    os.makedirs(DATA_PATH+folder)

all_imgs = os.listdir(IMG_PATH)
all_masks = os.listdir(MASK_PATH)

all_imgs.sort(key=lambda var:[int(x) if x.isdigit() else x 
                                for x in re.findall(r'[^0-9]|[0-9]+', var)])
all_masks.sort(key=lambda var:[int(x) if x.isdigit() else x 
                               for x in re.findall(r'[^0-9]|[0-9]+', var)])

random.seed(42)
random.shuffle(all_imgs)

train_split = int(0.8*len(all_imgs))
val_split = int(0.9*len(all_imgs))

train_imgs = all_imgs[:train_split]
val_imgs = all_imgs[train_split:val_split]
test_imgs = all_imgs[val_split:] 


train_masks = [f for f in all_masks if f in train_imgs]
val_masks = [f for f in all_masks if f in val_imgs]
test_masks = [f for f in all_masks if f in test_imgs]

def add_imgs(dir_name, image):
    img = Image.open(IMG_PATH+image)
    img.save(DATA_PATH+'/{}'.format(dir_name)+'/'+image)

def add_masks(dir_name, image):
    img = Image.open(MASK_PATH+image)
    img.save(DATA_PATH+'/{}'.format(dir_name)+'/'+image)    

img_folders = [(train_imgs, 'train_imgs'), (val_imgs, 'val_imgs'), (test_imgs, 'test_imgs')]
mask_folders = [(train_masks, 'train_masks'), (val_masks, 'val_masks'), (test_masks, 'test_masks')]

for folder in img_folders:
    array = folder[0]
    name = [folder[1]]*len(array)
    list(map(add_imgs, name, array))

for folder in mask_folders:
    array = folder[0]
    name = [folder[1]]*len(array)
    list(map(add_masks, name, array))