#doors & walls:
#background:0,0,0::
#thermal bridge:128,0,0::
#infil/exfil:0,128,0::
#physical defect:128,128,0::

#windows:
#background:0,0,0::
#thermal bridge:128,0,0::
#physical defect:0,128,0::
#infil/exfil:128,128,0::
import os
import torch
import random
import shutil
import numpy as np
from torch.utils import data
from torchvision import transforms

from PIL import Image, ImageOps, ImageFilter
import pdb

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}

class Normalize_isic(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask)
        mask = np.where(mask==255,1,0).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}

class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': mask}

class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}

class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}

class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask}

class AirBEM(data.Dataset):
    def __init__(self, base_dir, mode='default', shuffle=True, building='doors'):
        self.mode = mode #'default','train','val','train_val','test'
        self.shuffle = shuffle
        self.data_dir = os.path.join(base_dir, building) #'/data1/chenbin/AirBEM/dataset_semantic/doors'

        img_sets_txt = open(os.path.join(self.data_dir, 'ImageSets/Segmentation', self.mode+'.txt'))
        self.img_sets = img_sets_txt.readlines()
        self.img_sets = [line[:-1] for line in self.img_sets]
        if self.shuffle:
            random.shuffle(self.img_sets)

    def __len__(self):
        return len(self.img_sets)

    def __getitem__(self, idx):
        img_id = self.img_sets[idx] #'009938'
        img = Image.open( os.path.join(self.data_dir, 'JPEGImages', img_id+'.jpg') ) #PIL[h,w,3]
        try:
            mask = Image.open( os.path.join(self.data_dir, 'SegmentationClass', img_id+'.png') ) #PIL[h,w,3]
        except FileNotFoundError:  ## if failed, 
            np_img = np.array(img)
            h,w = np_img.shape[:-1]
            mask = np.zeros([h,w,3]).astype('uint8')
            mask = Image.fromarray(mask)
        
        img, mask = self._preprocess(img, mask)
        return img_id, img, mask
    
    def _preprocess(self, img, mask):
        composed_transforms = transforms.Compose([
            RandomHorizontalFlip(),
            RandomScaleCrop(base_size=513, crop_size=513),
            RandomRotate(180),
            RandomGaussianBlur(),
            ])
        sample = composed_transforms({'image': img, 'label': mask})
        return sample['image'], sample['label']

if __name__ == '__main__':
    base_dir = '/data1/chenbin/AirBEM/dataset_semantic'
    buildings = ['doors','walls','windows']
    for building in buildings:
        save_dir_img = os.path.join(base_dir, building, 'augmentation_img')
        save_dir_mask = os.path.join(base_dir, building, 'augmentation_mask')
        #if os.path.exists(save_dir_img):
        #    shutil.rmtree(save_dir_img)
        #os.makedirs(save_dir_img)
        #if os.path.exists(save_dir_mask):
        #    shutil.rmtree(save_dir_mask)
        #os.makedirs(save_dir_mask)

        dataset = AirBEM(base_dir, building=building)
        num = dataset.__len__()
        for i in range(num):
            imgID, img, mask = dataset.__getitem__(i)
            img.save( save_dir_img + '/' + imgID + '_augment3.jpg' )
            mask.save( save_dir_mask + '/' + imgID + '_augment3.png' )

