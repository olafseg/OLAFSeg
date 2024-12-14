from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
from data.pp_obj_part_mapping.map import aggregate_parts_to_classes
from utils.transforms import RandomHorizontalFlip, RandomScaleCrop, RandomGaussianBlur, Normalize, ToTensor, ResizeMasks

class SegmentationDataset(Dataset):
    def __init__(self, folder, mode='train', input_shape=(513, 513), num_classes=58):
        self.folder = folder
        self.mode = mode
        self.input_shape = input_shape
        self.num_classes = num_classes

        # Load paths
        with open(folder + mode + '.txt') as f:
            self.image_path_list = f.read().splitlines()

        # Only relevant for training mode
        if mode == 'train':
            self.anim_aggregation_map = aggregate_parts_to_classes(num_classes, animate=True, dataset = )
            self.inanim_aggregation_map = aggregate_parts_to_classes(num_classes, animate=False, dataset = )
            self.anim_classes = [3, 8, 10, 12, 13, 15, 17]
            self.inanim_classes = [1, 2, 5, 6, 7, 14, 16, 20]

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, i):
        # File paths
        image_path = self.folder + 'JPEGImages/' + self.image_path_list[i] + '.jpg'
        part_label_path = self.folder + 'GT_part_58/' + self.image_path_list[i] + '.png'
        obj_label_path = self.folder + 'object/' + self.image_path_list[i] + '.png'
        fb_label_path = self.folder + 'fb_from_obj_58/' + self.image_path_list[i] + '.png'
        edge_label_path = self.folder + 'hed_edges_58_2/' + self.image_path_list[i] + '.png'
        
        # Load images and initialize sample dictionary
        sample = {'image': Image.open(image_path)}
        org_size = sample['image'].size

        if self.mode == 'train':
            # Load labels and apply aggregation for training mode
            part_label = np.array(Image.open(part_label_path))
            sample['anim'] = Image.fromarray(self.aggregate_anim_labels(part_label))
            sample['inanim'] = Image.fromarray(self.aggregate_inanim_labels(part_label))

            obj_label = np.array(Image.open(obj_label_path))
            sample['anim_obj'] = Image.fromarray(self.anim_remove_objs(obj_label))
            sample['inanim_obj'] = Image.fromarray(self.inanim_remove_objs(obj_label))
            
        else:  # Inference mode
            sample['part'] = Image.open(part_label_path)
            sample['obj'] = Image.open(obj_label_path)

        # Shared labels and transformations for both modes
        sample['fb'] = Image.open(fb_label_path)
        sample['edge'] = Image.open(edge_label_path)
        sample['path'] = self.image_path_list[i]
        sample['orgsize'] = org_size

        # Apply transformations
        if self.mode == 'train':
            sample = self.transform_tr(sample)
        else:
            sample['org_img'] = np.array(sample['image'].copy())  # Save original image for inference mode
            sample['name'] = self.image_path_list[i] + '.png'
            sample = self.transform_val(sample)

        return sample

    # Methods for training mode only
    def anim_remove_objs(self, obj_label):
        final_label = np.zeros(obj_label.shape)
        for i in self.anim_classes:
            obj = (obj_label == i).astype(float)
            final_label += (obj * i)
        return final_label
    
    def inanim_remove_objs(self, obj_label):
        final_label = np.zeros(obj_label.shape)
        for i in self.inanim_classes:
            obj = (obj_label == i).astype(float)
            final_label += (obj * i)
        return final_label

    def aggregate_anim_labels(self, part_label):
        final_label = np.zeros(part_label.shape)
        for i in range(self.num_classes):
            part = (part_label == i).astype(float)
            final_label += self.anim_aggregation_map[i] * part
        return final_label
    
    def aggregate_inanim_labels(self, part_label):
        final_label = np.zeros(part_label.shape)
        for i in range(self.num_classes):
            part = (part_label == i).astype(float)
            final_label += self.inanim_aggregation_map[i] * part
        return final_label

    # Transformations for train and val/inference modes
    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            RandomHorizontalFlip(),
            RandomScaleCrop(base_size=513, crop_size=513),
            RandomGaussianBlur(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()])
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            ResizeMasks(crop_size=770),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()])
        return composed_transforms(sample)
