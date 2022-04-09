import torch
from ReadDataset import ReadDataset
from NeuralNetworks import LYTNet
from torch.utils.data import DataLoader
from loss import my_loss
import time
from helpers import direction_performance

test_file_dir = './Annotations/testing_file.csv'
img_dir = "./dataset"
save_path = 'TrainingModel/'

model_path = 'TrainingModel/_final_weights'

dataset = ReadDataset(csv_file=test_file_dir, img_dir=img_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

model = LYTNet()
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)
model.eval()

loss_fn = my_loss

#storing data
running_loss = 0
running_test_angle = 0
running_test_start = 0
running_test_end = 0

total = 0

start_time = time.time()

with torch.no_grad():
    for i, data in enumerate(dataloader):
        images = data['image']
        points = data['points']

    
    pred_direc = model(images)

    loss = my_loss(pred_direc, points)

    running_loss += loss.item()
    angle, start, end = direction_performance(pred_direc, points)

    running_test_angle += angle
    running_test_start += start
    running_test_end += end
    total += 1

print("Average loss: " + str(running_loss/total))
print("Average angle error: " + str(running_test_angle/total))
print("Average startpoint error: " + str(running_test_start/total))
print("Average endpoint error: " + str(running_test_end/total))
print("Time Elapsed: " + str(time.time() - start_time))