from ReadDataset import ReadDataset
from torch.utils.data import DataLoader
from NeuralNetworks import *
from loss import my_loss
import torch
from helpers import direction_performance
import matplotlib.pyplot as plt
import time




BATCH_SIZE = 32
MAX_EPOCHS = 40
INIT_LR = 0.001
WEIGHT_DECAY = 0.005 #0.00005
LR_DROP_MILESTONES = [400,600]

train_file_dir = './Annotations/training_file.csv'
valid_file_dir = './Annotations/validation_file.csv'
img_dir = "./dataset"
save_path = 'TrainingModel/'

train_dataset = ReadDataset(csv_file = train_file_dir, img_dir = img_dir)
valid_dataset = ReadDataset(csv_file = valid_file_dir, img_dir = img_dir)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)


model = LYTNet()
optimizer = torch.optim.AdamW(model.parameters(), lr=INIT_LR, weight_decay = WEIGHT_DECAY)
total_step = len(train_dataloader)

loss_fn = my_loss

train_losses = [] #stores the overall training loss at each epoch
valid_losses = [] #stores the overall validation loss at each epoch
val_angles = [] #stores the average angle error of the network during validation at each epoch
val_start = [] #stores the average startpoint error of the network during validation at each epoch
val_end = [] #stores the average endpoint error of the network during validation at each epoch
times = [] #stores the cumulating computing time for each epoch

start_time = time.time()

for epoch in range(MAX_EPOCHS):
#Load in the data in batches using the train_loader object

    # *** TRAINING ***

    model.train()

    train_loss = 0.0
    train_count = 0

    for i, data in enumerate(train_dataloader):  
        # Move tensors to the configured device
        images = data['image']
        labels = data['points']
        
        # Forward pass
        pred_direc = model(images)
        loss = loss_fn(pred_direc, labels)
        
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_count += 1
        train_loss += loss.item()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, MAX_EPOCHS, loss.item()))
    print('Average training loss: ' + str(train_loss/(train_count)))
    print('Time Elapsed: {}'.format(str(time.time()- start_time)))
    times.append(time.time()- start_time)
    train_losses.append(train_loss/train_count)




    # *** VALIDATION ***

    val_loss = 0
    val_angle_error = 0
    val_start_error = 0
    val_end_error = 0
    val_count = 0

    with torch.no_grad():
        for i, data in enumerate(train_dataloader):
            images = data['image']
            labels = data['points']
            pred_direc = model(images)
            val_count +=1
            val_loss += loss.item()
            
            angle, start, end = direction_performance(pred_direc, labels)
            val_angle_error += angle
            val_start_error += start
            val_end_error += end
    
    print("Average validation loss: " + str(val_loss/val_count))
    print("Angle Error: " + str(val_angle_error/val_count))
    print("Startpoint Error: " + str(val_start_error/val_count))
    print("Endpoint Error: " + str(val_end_error/val_count))

    valid_losses.append(val_loss/val_count)
    val_angles.append(val_angle_error/val_count)
    val_start.append(val_start_error/val_count)
    val_end.append(val_end_error/val_count)
    
    #stores the network and optimizer weights every 10th epoch
    if epoch%10 == 9:
        states = {
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
                }
        torch.save(states, save_path + '_epoch_' + str(epoch+1))


################
#AFTER TRAINING#
################

#plots training and validation loss
plt.title('train and validation loss')
plt.plot(valid_losses)
plt.plot(train_losses)
plt.savefig(save_path + '_losses')
plt.show()

#plots training and validation loss
plt.title('Training tima by epoch')
plt.plot(times)
plt.savefig(save_path + '_training_time')
plt.show()

#save final network weights
torch.save(model.state_dict(), save_path + '_final_weights')