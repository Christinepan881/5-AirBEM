import os
import numpy as np 
import random
import pdb

base_dir = '/data1/chenbin/AirBEM/dataset_semantic'
buildings = ['doors','walls','windows']
img_dir = 'augmentation_img'
txt_dir = 'ImageSets/Segmentation'

for building in buildings:
    data_dir = os.path.join(base_dir, building, img_dir)
    train_txt = os.path.join(base_dir, building, txt_dir, 'train.txt' )
    val_txt = os.path.join(base_dir, building, txt_dir, 'val.txt')
    train_val_txt = os.path.join(base_dir, building, txt_dir, 'train_val.txt')
    test_txt = os.path.join(base_dir, building, txt_dir, 'test.txt')

    imgs_id = [f.split('.')[0] for f in os.listdir(data_dir) if f.endswith(".jpg")] #['FLIR0002_augmentation0']
    random.shuffle(imgs_id)
    num = len(imgs_id)
    num_train = int(0.6*num)
    num_val = int(0.2*num)

    id_train = imgs_id[:num_train]
    id_val = imgs_id[num_train:(num_train+num_val)]
    id_test = imgs_id[(num_train+num_val):]

    f_train = open(train_txt, 'w')
    f_val = open(val_txt, 'w')
    f_train_val = open(train_val_txt, 'w')
    f_test = open(test_txt, 'w')
    f_train.write('\n'.join(id_train))
    f_val.write('\n'.join(id_val))
    f_train_val.write('\n'.join(id_train + id_val))
    f_test.write('\n'.join(id_test))
    f_train.close() # = open(train_txt, 'w')
    f_val.close() # = open(val_txt, 'w')
    f_train_val.close() # = open(train_val_txt, 'w')
    f_test.close() # = open(test_txt, 'w')