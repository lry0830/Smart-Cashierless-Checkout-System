import albumentations as A
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import cv2
import os
import imgaug as ia
import imgaug.augmenters as iaa
import math
import random
import copy
import glob
import argparse

class Augment:
    def __init__(self, images_dir, anno_dir, mask_dir, n, choice):
        self.images_path = images_dir
        self.anno_path = anno_dir
        self.mask_path = mask_dir
        self.N = int(n)
        self.imgtype = choice

        imgs_save_path = '/content/augmented/images'
        if not os.path.exists(imgs_save_path):
            os.makedirs(imgs_save_path)
        txt_save_path = '/content/augmented/txt'
        if not os.path.exists(txt_save_path):
            os.makedirs(txt_save_path)
        mask_save_path = '/content/augmented/masks'
        if not os.path.exists(mask_save_path):
            os.makedirs(mask_save_path)

        random.seed(7)

    def readImage(self, filename):
        print(filename)
        img = cv2.imread(filename)
        return img

    def readYolo(self,filename):
        coords = []
        with open(filename, 'r') as fname:
            for file1 in fname:
                x = file1.strip().split(' ')
                x.append(x[0])
                x.pop(0)
                x[0] = float(x[0])
                x[1] = float(x[1])
                x[2] = float(x[2])
                x[3] = float(x[3])
                coords.append(x)
        return coords

    def getTransform(self):
        try:
          if self.imgtype == "syn":
              transform = A.Compose([A.RandomRotate90(p=1), A.Resize(height=2300, width=2300, interpolation=cv2.INTER_AREA),
                                    A.ShiftScaleRotate(scale_limit=[0, 0], rotate_limit=45, p=1,
                                                        border_mode=cv2.BORDER_CONSTANT, mask_value=0, shift_limit=0),
                                    A.VerticalFlip(p=0), A.MotionBlur(blur_limit=5, p=0.8), A.RandomToneCurve(scale=0.5, p=0),
				    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.0),
                                    A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.0, p=0.5), ],
                                    bbox_params=A.BboxParams(format='yolo'))
          if self.imgtype == "real":
              transform = A.Compose([A.RandomRotate90(p=0.9), A.Resize(height=3200, width=3200, interpolation=cv2.INTER_AREA),
                                   A.ShiftScaleRotate(scale_limit=[0, 0], rotate_limit=3, p=1,
                                                      border_mode=cv2.BORDER_REFLECT_101, shift_limit=0),
                                   A.VerticalFlip(p=0), A.MotionBlur(blur_limit=5, p=0.8), A.RandomToneCurve(scale=0.25, p=1),
				   A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.25), p=0.5),
                                   A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0, p=0.5), ], bbox_params=A.BboxParams(format='yolo'))

          return transform
        except:
          raise Exception('Invalid Argument')
        

    def writeYolo(self, coords, count, name):

        with open(f'/content/augmented/txt/{name}({count}).txt', "w") as f:
            for x in coords:
                f.write("%s %s %s %s %s \n" % (x[-1], x[0], x[1], x[2], x[3]))

    def start(self):
        imagespath = self.images_path
        maskpath = self.mask_path
        txtpath = self.anno_path
        max_iter = self.N
        bboxes = None
        mask = None

        for filename in sorted(os.listdir(imagespath)):

            if filename.endswith(".jpg") or filename.endswith(".JPG"):
                title, ext = os.path.splitext(os.path.basename(filename))
                image = Augment.readImage(self,f"{imagespath}/{filename}")
                img = copy.deepcopy(image)

                # For synthesized image
                if self.imgtype == 'syn':
                   annoname = title + '.txt'
                   maskname = title + '.png'
                   if maskname in os.listdir(maskpath):
                       print(f'{maskpath}/{maskname}')
                       mask = cv2.imread(f'{maskpath}/{maskname}')
                   if annoname in os.listdir(txtpath):
                       bboxes = Augment.readYolo(self, f'{txtpath}/{annoname}')
                elif self.imgtype == 'real':
                    annoname = title + '.txt'
                    if annoname in os.listdir(txtpath):
                       bboxes = Augment.readYolo(self, f'{txtpath}/{annoname}')

                for count in range(1, max_iter+1):
                    transform = Augment.getTransform(self)
                    name = title + '(' + str(count) + ')' + '.jpg'
                    if self.imgtype == 'syn':
                      transformed = transform(image=img, bboxes=bboxes, mask=mask)
                      transformed_bboxes = transformed['bboxes']
                      transformed_mask = transformed['mask']
                      cv2.imwrite(f'/content/augmented/masks/{name}', transformed_mask)
                      Augment.writeYolo(self, transformed_bboxes, count, title)
                    elif self.imgtype == 'real':
                      transformed = transform(image=img, bboxes=bboxes)
                      transformed_bboxes = transformed['bboxes']
                      Augment.writeYolo(self, transformed_bboxes, count, title)

                    transformed_image = transformed['image']
                    cv2.imwrite(f'/content/augmented/images/{name}', transformed_image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Augmentation')
    parser.add_argument('-N', '--generated_num', help='Number of generated images', required=True)
    parser.add_argument('-i', '--images_dir', help='Images directory', required=True)
    parser.add_argument('-a', '--anno_dir', help='Annotation directory', required=True)
    parser.add_argument('-m', '--mask_dir',help='Mask directory', required=True)
    parser.add_argument('-t', '--img_type', help='Image type', required=True)
    args = parser.parse_args()

    session = Augment(args.images_dir, args.anno_dir, args.mask_dir, args.generated_num, args.img_type)
    session.start()

