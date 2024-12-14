import torch
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter


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
        anim_obj = sample['anim_obj']
        inanim_obj = sample['inanim_obj']
        anim = sample['anim']
        inanim = sample['inanim']
        fb = sample['fb']
        edge = sample['edge']

        img = np.array(img).astype(np.float32)
        anim_obj = np.array(anim_obj).astype(np.float32)
        inanim_obj = np.array(inanim_obj).astype(np.float32)
        anim = np.array(anim).astype(np.float32)
        inanim = np.array(inanim).astype(np.float32)
        fb = np.array(fb).astype(np.float32)
        edge = np.array(edge).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'anim_obj': anim_obj,
                'inanim_obj': inanim_obj,
                'anim': anim,
                'inanim': inanim,
                'fb': fb,
                'edge': edge}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        anim_obj = sample['anim_obj']
        inanim_obj = sample['inanim_obj']
        anim = sample['anim']
        inanim = sample['inanim']
        fb = sample['fb']
        edge = sample['edge']

        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        anim_obj = np.array(anim_obj).astype(np.float32)
        inanim_obj = np.array(inanim_obj).astype(np.float32)
        anim = np.array(anim).astype(np.float32)
        inanim = np.array(inanim).astype(np.float32)
        fb = np.array(fb).astype(np.float32)
        edge = np.array(edge).astype(np.float32)
        


        img = np.vstack([img, fb[None, :, :]])
        img = np.vstack([img, edge[None, :, :]])

        #print(np.shape(img))

        img = torch.from_numpy(img).float()
        anim_obj = torch.from_numpy(anim_obj).float()
        inanim_obj = torch.from_numpy(inanim_obj).float()
        anim = torch.from_numpy(anim).float()
        inanim = torch.from_numpy(inanim).float()

        return {'image': img,
                'anim_obj': anim_obj,
                'inanim_obj': inanim_obj,
                'anim': anim,
                'inanim': inanim}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        anim_obj = sample['anim_obj']
        inanim_obj = sample['inanim_obj']
        anim = sample['anim']
        inanim = sample['inanim']
        fb = sample['fb']
        edge = sample['edge']

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            anim_obj = anim_obj.transpose(Image.FLIP_LEFT_RIGHT)
            inanim_obj = inanim_obj.transpose(Image.FLIP_LEFT_RIGHT)
            anim = anim.transpose(Image.FLIP_LEFT_RIGHT)
            inanim = inanim.transpose(Image.FLIP_LEFT_RIGHT)
            fb = fb.transpose(Image.FLIP_LEFT_RIGHT)
            edge = edge.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'anim_obj': anim_obj,
                'inanim_obj': inanim_obj,
                'anim': anim,
                'inanim': inanim,
                'fb': fb,
                'edge': edge}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        anim_obj = sample['anim_obj']
        inanim_obj = sample['inanim_obj']
        anim = sample['anim']
        inanim = sample['inanim']

        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        anim_obj = anim_obj.rotate(rotate_degree, Image.NEAREST)
        inanim_obj = inanim_obj.rotate(rotate_degree, Image.NEAREST)
        anim = anim.rotate(rotate_degree, Image.NEAREST)
        inanim = inanim.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'anim_obj': anim_obj,
                'inanim_obj': inanim_obj,
                'anim': anim,
                'inanim': inanim}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        anim_obj = sample['anim_obj']
        inanim_obj = sample['inanim_obj']
        anim = sample['anim']
        inanim = sample['inanim']
        fb = sample['fb']
        edge = sample['edge']

        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'anim_obj': anim_obj,
                'inanim_obj': inanim_obj,
                'anim': anim,
                'inanim': inanim,
                'fb': fb,
                'edge': edge}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        anim_obj = sample['anim_obj']
        inanim_obj = sample['inanim_obj']
        anim = sample['anim']
        inanim = sample['inanim']
        fb = sample['fb']
        edge = sample['edge']

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
        anim_obj = anim_obj.resize((ow, oh), Image.NEAREST)
        inanim_obj = inanim_obj.resize((ow, oh), Image.NEAREST)
        anim = anim.resize((ow, oh), Image.NEAREST)
        inanim = inanim.resize((ow, oh), Image.NEAREST)
        fb = fb.resize((ow, oh), Image.NEAREST)
        edge = edge.resize((ow, oh), Image.NEAREST)

        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            anim_obj = ImageOps.expand(anim_obj, border=(0, 0, padw, padh), fill=self.fill)
            inanim_obj = ImageOps.expand(inanim_obj, border=(0, 0, padw, padh), fill=self.fill)
            anim = ImageOps.expand(anim, border=(0, 0, padw, padh), fill=self.fill)
            inanim = ImageOps.expand(inanim, border=(0, 0, padw, padh), fill=self.fill)
            fb = ImageOps.expand(fb, border=(0, 0, padw, padh), fill=self.fill)
            edge = ImageOps.expand(edge, border=(0, 0, padw, padh), fill=self.fill)

        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        anim_obj = anim_obj.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        inanim_obj = inanim_obj.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        anim = anim.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        inanim = inanim.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        fb = fb.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        edge = edge.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))



        return {'image': img,
                'anim_obj': anim_obj,
                'inanim_obj': inanim_obj,
                'anim': anim,
                'inanim': inanim,
                'fb': fb,
                'edge': edge}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        anim_obj = sample['anim_obj']
        inanim_obj = sample['inanim_obj']
        anim = sample['anim']
        inanim = sample['inanim']

        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        anim_obj = anim_obj.resize((ow, oh), Image.NEAREST)
        inanim_obj = inanim_obj.resize((ow, oh), Image.NEAREST)
        anim = anim.resize((ow, oh), Image.NEAREST)
        inanim = inanim.resize((ow, oh), Image.NEAREST)

        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        anim_obj = anim_obj.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        inanim_obj = inanim_obj.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        anim = anim.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        inanim = inanim.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'anim_obj': anim_obj,
                'inanim_obj': inanim_obj,
                'anim': anim,
                'inanim': inanim}

class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        anim_obj = sample['anim_obj']
        inanim_obj = sample['inanim_obj']
        anim = sample['anim']
        inanim = sample['inanim']

        # assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        anim_obj = anim_obj.resize(self.size, Image.NEAREST)
        inanim_obj = inanim_obj.resize(self.size, Image.NEAREST)
        anim = anim.resize(self.size, Image.NEAREST)
        inanim = inanim.resize(self.size, Image.NEAREST)

        return {'image': img,
                'anim_obj': anim_obj,
                'inanim_obj': inanim_obj,
                'anim': anim,
                'inanim': inanim}
    
class ResizeMasks(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        anim_obj = sample['anim_obj']
        inanim_obj = sample['inanim_obj']
        anim = sample['anim']
        inanim = sample['inanim']

        w, h = img.size
        short_size = 0
        if w > h:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
            short_size = oh
        else:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
            short_size = ow

        img = img.resize((ow, oh), Image.BILINEAR)
        anim_obj = anim_obj.resize((ow, oh), Image.NEAREST)
        inanim_obj = inanim_obj.resize((ow, oh), Image.NEAREST)
        anim = anim.resize((ow, oh), Image.NEAREST)
        inanim = inanim.resize((ow, oh), Image.NEAREST)

        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            anim_obj = ImageOps.expand(anim_obj, border=(0, 0, padw, padh), fill=0)
            inanim_obj = ImageOps.expand(inanim_obj, border=(0, 0, padw, padh), fill=0)
            anim = ImageOps.expand(anim, border=(0, 0, padw, padh), fill=0)
            inanim = ImageOps.expand(inanim, border=(0, 0, padw, padh), fill=0)

        return {'image': img,
                'anim_obj': anim_obj,
                'inanim_obj': inanim_obj,
                'anim': anim,
                'inanim': inanim}
