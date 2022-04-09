import pandas as pd
from torch.utils.data import Dataset
import os
from PIL import Image, ImageFile
import torch
from torchvision import transforms

class ReadDataset(Dataset):
    def __init__(self, csv_file, img_dir, transformation = True):
        self.labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transformation = transformation
        
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img_name = os.path.join(self.img_dir, self.labels.iloc[index, 0])
        image = Image.open(img_name)
        
        points = self.labels.iloc[index, 2:6]
        points = [points[0]/4032, points[1]/3024, points[2]/4032, points[3]/3024]
        points = torch.FloatTensor(points)
        
        model_mean = [0.485, 0.456, 0.406]
        model_std = [0.229, 0.224, 0.225]

        input_image_height = 576
        input_image_width = 768
        
        data_transforms = transforms.Compose([
            transforms.Resize((input_image_height, input_image_width)),
            transforms.ToTensor(),
            #transforms.ColorJitter(0.05,0.05,0.05,0.01),
            transforms.Normalize(model_mean, model_std)])
        
        image = data_transforms(image)
        
        label = {'image': image, 'points': points}
              
        return label