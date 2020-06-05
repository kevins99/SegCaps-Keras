import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import numpy as np
import skimage.io as io
import os 
from losses import loss
from metrics import dice_coef
import argparse
from capsnet import CapsNetR3

from generator import train_generator, val_generator


Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255



def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.6
    sess = tf.Session(config=config)
    KTF.set_session(sess)

    NO_OF_TRAINING_IMAGES = len(os.listdir('./data/train_imgs/train/'))
    NO_OF_VAL_IMAGES = len(os.listdir('./data/val_imgs/val/'))

    NO_OF_EPOCHS = 60
    BATCH_SIZE = 1

    BASE_DIR = os.getcwd()
    WEIGHTS_PATH = BASE_DIR+'/weights/weights.h5'

    model = CapsNetR3((256, 256, 3))
    opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(loss=loss, 
                    optimizer=opt, 
                    metrics=[dice_coef, 'accuracy'])

    model.summary()
    print(model.metrics_names)
    checkpoint = ModelCheckpoint(WEIGHTS_PATH, monitor='val_dice_coef',
                                verbose=1, save_best_only=True, mode='max')

    csv_logger = CSVLogger('./log.out', append=True, separator=';')

    earlystopping = EarlyStopping(monitor = 'val_loss', verbose = 1,
                                min_delta = 0.01, patience = 3, mode = 'max')

    tensorboard_callback = TensorBoard(log_dir="./logs")

    callbacks_list = [checkpoint, csv_logger, tensorboard_callback]

    results = model.fit_generator(train_generator, epochs=NO_OF_EPOCHS, 
                            steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                            validation_data=val_generator, 
                            validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE),
                            callbacks=callbacks_list)

    from generator import test_generator

    filenames = test_generator.filenames
    nb_samples = len(filenames)

    results = model.predict_generator(test_generator, steps=nb_samples)
    saveResult("./results",results)





