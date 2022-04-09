from Model.NeuralNetworks import *
import torch
import torch.nn as nn

from torchvision import transforms
import coremltools as ct
from PIL import Image

### This file is used to convert our Pytorch model into a coreML model for the integration into an IOS app

# Load model
model_path = 'TrainingModel/_final_weights'

model = LYTNet()
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)
model.eval()

#Load sample image
input_image = Image.open('dataset/john_IMG_2193.jpg')

model_mean = [0.485, 0.456, 0.406]
model_std = [0.229, 0.224, 0.225]

input_image_height = 576
input_image_width = 768

data_transforms = transforms.Compose([
            transforms.Resize((input_image_height, input_image_width)),
            transforms.ToTensor()])

input_tensor = data_transforms(input_image)
input_batch = input_tensor.unsqueeze(0)

#Trace the Model with Pytorch
trace = torch.jit.trace(model, input_batch)

#Convert to Core ML

#Define input
_input = ct.ImageType(
    name="input_1",
    shape=input_batch.shape,
    bias=[- 0.485/(0.229) , - 0.456/(0.224), - 0.406/(0.225)],
    scale=1/(0.226*255.0)
)

#Convert model
mlmodel = ct.convert(
    trace,
    inputs=[_input]
)

#Save the model
mlmodel.save("CrosswalkVisionModel.mlmodel")

