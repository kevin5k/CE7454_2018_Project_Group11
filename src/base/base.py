import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import randint
import time
from ships_loader import create_imdb, ships_dataset
import signal, os
import numpy as np
import cv2
import matplotlib.pyplot as plt 

device= torch.device("cuda")
#device= torch.device("cpu")
print(device)

parser = argparse.ArgumentParser()

parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      action='store_true')

parser.add_argument('--d', dest='demo',
                      help='demo on some test set',
                      action='store_true')

parser.add_argument('--e', dest='eval',
                      help='evaluate on full test set',
                      action='store_true')


parser.add_argument('--t', dest='train',
                      help='train on train set',
                      action='store_true')

args = parser.parse_args()

args.eval == True
args.train == False

class VGG(nn.Module):

    def __init__(self):

        super(VGG, self).__init__()

        # block 1:         3 x 32 x 32 --> 64 x 16 x 16        
        self.conv1a = nn.Conv2d(3,   64,  kernel_size=3, padding=1 )
        self.conv1b = nn.Conv2d(64,  64,  kernel_size=3, padding=1 )
        self.bn1a = nn.BatchNorm2d(64)
        self.bn1b = nn.BatchNorm2d(64)
        self.pool1  = nn.MaxPool2d(2,2)

        # block 2:         64 x 16 x 16 --> 128 x 8 x 8
        self.conv2a = nn.Conv2d(64,  128, kernel_size=3, padding=1 )
        self.conv2b = nn.Conv2d(128, 128, kernel_size=3, padding=1 )
        self.bn2a = nn.BatchNorm2d(128)
        self.bn2b = nn.BatchNorm2d(128)
        self.pool2  = nn.MaxPool2d(2,2)

        # block 3:         128 x 8 x 8 --> 256 x 4 x 4        
        self.conv3a = nn.Conv2d(128, 256, kernel_size=3, padding=1 )
        self.conv3b = nn.Conv2d(256, 256, kernel_size=3, padding=1 )
        self.bn3a = nn.BatchNorm2d(256)
        self.bn3b = nn.BatchNorm2d(256)
        self.pool3  = nn.MaxPool2d(2,2)
        
        #block 4:          256 x 4 x 4 --> 512 x 2 x 2
        self.conv4a = nn.Conv2d(256, 512, kernel_size=3, padding=1 )
        self.bn4a = nn.BatchNorm2d(512)
        self.pool4  = nn.MaxPool2d(2,2)

        self.conv5a = nn.Conv2d(512, 512, kernel_size=3, padding=1 )
        self.bn5a = nn.BatchNorm2d(512)
        self.pool5  = nn.MaxPool2d(2,2)

        self.conv6a = nn.Conv2d(512, 512, kernel_size=3, padding=1 )
        self.bn6a = nn.BatchNorm2d(512)
        self.pool6  = nn.MaxPool2d(2,2)

        # linear layers:   512 x 2 x 2 --> 2048 --> 4096 --> 4096 --> 10
        self.linear1 = nn.Linear(9 * 9 * 512, 4096)
        self.linear2 = nn.Linear(4096,4096)
        self.linear3 = nn.Linear(4096, 2)


    def forward(self, x):

        # block 1:         3 x 32 x 32 --> 64 x 16 x 16
        x = self.conv1a(x)
        x = self.bn1a(x)
        x = F.relu(x)
        x = self.conv1b(x)
        x = self.bn1b(x)
        x = F.relu(x)
        x = self.pool1(x)

        # block 2:         64 x 16 x 16 --> 128 x 8 x 8
        x = self.conv2a(x)
        x = self.bn2a(x)
        x = F.relu(x)
        x = self.conv2b(x)
        x = self.bn2b(x)
        x = F.relu(x)
        x = self.pool2(x)

        # block 3:         128 x 8 x 8 --> 256 x 4 x 4
        x = self.conv3a(x)
        x = self.bn3a(x)
        x = F.relu(x)
        x = self.conv3b(x)
        x = self.bn3b(x)
        x = F.relu(x)
        x = self.pool3(x)

        #block 4:          256 x 4 x 4 --> 512 x 2 x 2
        x = self.conv4a(x)
        x = self.bn4a(x)
        x = F.relu(x)
        x = self.pool4(x)

        #block 5:          256 x 4 x 4 --> 512 x 2 x 2
        x = self.conv5a(x)
        x = self.bn5a(x)
        x = F.relu(x)
        x = self.pool5(x)

        #block 6:          256 x 4 x 4 --> 512 x 2 x 2
        x = self.conv6a(x)
        x = self.bn6a(x)
        x = F.relu(x)
        x = self.pool6(x)

        # linear layers:   512 x 2 x 2 --> 2048 --> 4096 --> 4096 --> 10
        x = x.view(-1, 9 * 9 * 512)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x) 
        
        return x


# ### Build the net. How many parameters in total? (the one layer net had 30,000 parameters)

# In[7]:

my_lr=0.002 
net = VGG()
start_epoch = 1
start_step = 1

if args.resume or args.demo or args.eval:
    #net.load_state_dict(torch.load("./saved_model.pth"))
    checkpoint = torch.load("../../models/base/vgg_base_model.pth")
    start_epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint['model'])
    my_lr = checkpoint['lr']
    start_step = checkpoint['step']


print(net)


net = net.to(device)

criterion = nn.CrossEntropyLoss()


def get_error( scores , labels ):

    bs=scores.size(0)
    predicted_labels = scores.argmax(dim=1)
    indicator = (predicted_labels == labels)
    num_matches=indicator.sum()
    
    return 1-num_matches.float()/bs, num_matches

file_dir = ""
image_dir = ""
if args.demo or args.eval:
    file_dir = "../../dataset/input/test_ship_segmentations.csv"
    image_dir = "../../dataset/test"
elif args.train:
    file_dir = "../../dataset/input/train_ship_segmentations.csv"
    image_dir = "../../dataset/train"
else:
    print("Please provide input data info")
    exit()

imdb = create_imdb(file_dir, image_dir)
dataset = ships_dataset(imdb)
bs = 4

if args.demo:
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle=True)
    data_iter = iter(dataloader)
    for i in range(10):
        inputs, labels, image_files = next(data_iter)
        inputs=inputs.to(device)
        scores=net( inputs )
        prob = F.softmax(scores)
        im = cv2.imread(image_files[0])
        values, indices = torch.max(prob, 1)
        #print(indices)
        #print(labels)
        text = "Predict: " + ("has ship" if indices[0] == 1 else "no ship") + "   GT: " + ("has ship" if labels[0] == 1 else "no ship")
        cv2.putText(im, text, (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3, lineType=cv2.LINE_AA)
        plt.imshow(im)
        plt.show()
    exit()


if args.eval:
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle=True)
    data_iter = iter(dataloader)
    num_matched=0
    num_batches=0
    data_size = len(data_iter)
    for i in range(len(data_iter)):
        inputs, labels, image_files = next(data_iter)
        inputs=inputs.to(device)
        labels = labels.to(device)
        scores = net( inputs )
        #prob = F.softmax(scores)
        
        # compute the error made on this batch and add it to the running error       
        error, matched = get_error( scores.detach() , labels)
        num_matched += matched.item()
        num_batches += 1
        if i % 100 == 0:
            print("(%s/%d)\t: %.4lf%%" % (i, data_size, (num_matched / num_batches) * 100))
    exit()


train_size = len(dataset)
step = int(train_size / bs)
print("train size: " + str(train_size))
print("step: " + str(step))

dataloader = torch.utils.data.DataLoader(dataset, batch_size = bs, shuffle=True)
start=time.time()

lost = []
errors = []

for epoch in range(start_epoch, 20):
    data_iter = iter(dataloader)
    # divide the learning rate by 2 at epoch 10, 14 and 18
    
    # create a new optimizer at the beginning of each epoch: give the current learning rate.   
    optimizer=torch.optim.SGD( net.parameters() , lr=my_lr )
        
    # set the running quatities to zero at the beginning of the epoch
    # running_loss=0
    # running_error=0
    # num_batches=0
 
    for count in range(start_step, step + 1):
        # set the running quatities to zero at the beginning of the epoch
        running_loss=0
        running_error=0
        num_batches=0

        # Set the gradients to zeros
        optimizer.zero_grad()
        
        # create a minibatch       
        minibatch_data, minibatch_label, _ = next(data_iter)
        
        # send them to the gpu
        minibatch_data=minibatch_data.to(device)
        minibatch_label=minibatch_label.to(device)
        
        inputs = minibatch_data
        # tell Pytorch to start tracking all operations that will be done on "inputs"
        inputs.requires_grad_()

        # forward the minibatch through the net 
        scores=net( inputs ) 

        # Compute the average of the losses of the data points in the minibatch
        loss =  criterion( scores , minibatch_label) 
        
        # backward pass to compute dL/dU, dL/dV and dL/dW   
        loss.backward()

        # do one step of stochastic gradient descent: U=U-lr(dL/dU), V=V-lr(dL/dU), ...
        optimizer.step()
        

        # START COMPUTING STATS
        
        # add the loss of this batch to the running loss
        running_loss += loss.detach().item()
        
        # compute the error made on this batch and add it to the running error       
        error, _ = get_error( scores.detach() , minibatch_label)
        running_error += error.item()
        
        num_batches+=1        
    
        if(count % 100 == 0):
            
            # compute stats for the full training set
            total_loss = running_loss/num_batches
            total_error = running_error/num_batches
            elapsed = (time.time()-start)/60

            if int(elapsed) % 10 == 0:
                lost.append(total_loss)
                errors.append(total_error)
                torch.save({
                    'model': net.state_dict(),
                    'epoch': epoch,
                    'step': count,
                    'lr': my_lr,
                    'lost': lost,
                    'error':error
                }, "../../models/base/vgg_base_model.pth")
            

            print('epoch=',epoch, '\t step=', count, '/', step, '\t time=', elapsed,'min','\t lr=', my_lr  ,'\t loss=', total_loss , '\t error=', total_error*100 ,'percent')
            #eval_on_test_set() 
            print(' ')
            
    my_lr = my_lr / 2




