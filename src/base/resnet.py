
# coding: utf-8

# # Lab 10.04 : VGG architecture - solution

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import randint
import utils
import time


# ### With or without GPU?
# 
# It is recommended to run this code on GPU:<br> 
# * Time for 1 epoch on CPU : 841 sec (14.02 min)<br> 
# * Time for 1 epoch on GPU : 9 sec w/ GeForce GTX 1080 Ti <br>

# In[2]:


device= torch.device("cuda")
#device= torch.device("cpu")
print(device)


# ### Download the CIFAR dataset 

# In[3]:


from utils import check_cifar_dataset_exists
data_path=check_cifar_dataset_exists()

train_data=torch.load(data_path+'cifar/train_data.pt')
train_label=torch.load(data_path+'cifar/train_label.pt')
test_data=torch.load(data_path+'cifar/test_data.pt')
test_label=torch.load(data_path+'cifar/test_label.pt')

print(train_data.size())
print(test_data.size())


# ### Compute mean pixel intensity over all training set and all channels

# In[4]:


mean= train_data.mean()

print(mean)


# ### Compute standard deviation

# In[5]:


std= train_data.std()

print(std)


# ### Make a VGG convnet class. 

# In[6]:


class VGG_convnet(nn.Module):

    def __init__(self):

        super(VGG_convnet, self).__init__()

        # block 1:         3 x 32 x 32 --> 64 x 16 x 16        
        self.conv1a = nn.Conv2d(3,   64,  kernel_size=3, padding=1, bias=False )
        self.conv1b = nn.Conv2d(64,  64,  kernel_size=3, padding=1, bias=False )
        self.bn1a = nn.BatchNorm2d(64)
        self.bn1b = nn.BatchNorm2d(64)
        self.pool1  = nn.MaxPool2d(2,2)

        # block 2:         64 x 16 x 16 --> 128 x 8 x 8
        self.conv2a = nn.Conv2d(64,  128, kernel_size=3, padding=1, bias=False )
        self.conv2b = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False )
        self.bn2a = nn.BatchNorm2d(128)
        self.bn2b = nn.BatchNorm2d(128)
        self.resize2a = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.resize2b = nn.BatchNorm2d(128)
        self.pool2  = nn.MaxPool2d(2,2)

        # block 3:         128 x 8 x 8 --> 256 x 4 x 4        
        self.conv3a = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False )
        self.conv3b = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False )
        self.bn3a = nn.BatchNorm2d(256)
        self.bn3b = nn.BatchNorm2d(256)
        self.resize3a = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.resize3b = nn.BatchNorm2d(256)
        self.pool3  = nn.MaxPool2d(2,2)
        
        #block 4:          256 x 4 x 4 --> 512 x 2 x 2
        self.conv4a = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False )
        self.bn4a = nn.BatchNorm2d(512)
        self.resize4a = nn.Conv2d(256, 512, kernel_size=1, bias=False)
        self.resize4b = nn.BatchNorm2d(512)
        self.pool4  = nn.MaxPool2d(2,2)

        # linear layers:   512 x 2 x 2 --> 2048 --> 4096 --> 4096 --> 10
        self.linear1 = nn.Linear(2048, 4096)
        self.linear2 = nn.Linear(4096,4096)
        self.linear3 = nn.Linear(4096, 10)


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
        residual = x
        residual = self.resize2a(residual)
        residual = self.resize2b(residual)
        x = self.conv2a(x)
        x = self.bn2a(x)
        x = F.relu(x)
        x = self.conv2b(x)
        x = self.bn2b(x)
        x = x + residual
        x = F.relu(x)
        x = self.pool2(x)

        # block 3:         128 x 8 x 8 --> 256 x 4 x 4
        residual = x
        residual = self.resize3a(residual)
        residual = self.resize3b(residual)
        x = self.conv3a(x)
        x = self.bn3a(x)
        x = F.relu(x)
        x = self.conv3b(x)
        x = self.bn3b(x)
        x = x + residual
        x = F.relu(x)
        x = self.pool3(x)

        #block 4:          256 x 4 x 4 --> 512 x 2 x 2
        residual = x
        residual = self.resize4a(residual)
        residual = self.resize4b(residual)
        x = self.conv4a(x)
        x = self.bn4a(x)
        x = x + residual
        x = F.relu(x)
        x = self.pool4(x)

        # linear layers:   512 x 2 x 2 --> 2048 --> 4096 --> 4096 --> 10
        x = x.view(-1, 2048)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x) 
        
        return x


# ### Build the net. How many parameters in total? (the one layer net had 30,000 parameters)

# In[7]:


net=VGG_convnet()

print(net)
utils.display_num_param(net)


# ### Send the weights of the networks to the GPU (as well as the mean and std)

# In[8]:


net = net.to(device)

mean = mean.to(device)

std = std.to(device)


# ### Choose the criterion, batch size, and initial learning rate. Select the following:
# * batch size =128
# * initial learning rate =0.25
# 

# In[9]:


criterion = nn.CrossEntropyLoss()
my_lr=0.25 
bs= 128


# ### Function to evaluate the network on the test set (NOTHING TO DO HERE)

# In[10]:


def eval_on_test_set():

    running_error=0
    num_batches=0

    for i in range(0,10000,bs):

        minibatch_data =  test_data[i:i+bs]
        minibatch_label= test_label[i:i+bs]

        minibatch_data=minibatch_data.to(device)
        minibatch_label=minibatch_label.to(device)
        
        inputs = (minibatch_data - mean)/std

        scores=net( inputs ) 

        error = utils.get_error( scores , minibatch_label)

        running_error += error.item()

        num_batches+=1

    total_error = running_error/num_batches
    print( 'error rate on test set =', total_error*100 ,'percent')


# ### Do 20 passes through the training set. Divide the learning rate by 2 at epoch 10, 14 and 18.

# In[ ]:


start=time.time()

for epoch in range(1,20):
    
    # divide the learning rate by 2 at epoch 10, 14 and 18
    if epoch==10 or epoch == 14 or epoch==18:
        my_lr = my_lr / 2
    
    # create a new optimizer at the beginning of each epoch: give the current learning rate.   
    optimizer=torch.optim.SGD( net.parameters() , lr=my_lr , weight_decay=0.0001)
        
    # set the running quatities to zero at the beginning of the epoch
    running_loss=0
    running_error=0
    num_batches=0
    
    # set the order in which to visit the image from the training set
    shuffled_indices=torch.randperm(50000)
 
    for count in range(0,50000,bs):
    
        # Set the gradients to zeros
        optimizer.zero_grad()
        
        # create a minibatch       
        indices=shuffled_indices[count:count+bs]
        minibatch_data =  train_data[indices]
        minibatch_label=  train_label[indices]
        
        # send them to the gpu
        minibatch_data=minibatch_data.to(device)
        minibatch_label=minibatch_label.to(device)
        
        # normalize the minibatch (this is the only difference compared to before!)
        inputs = (minibatch_data - mean)/std
        
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
        error = utils.get_error( scores.detach() , minibatch_label)
        running_error += error.item()
        
        num_batches+=1        
    
    
    # compute stats for the full training set
    total_loss = running_loss/num_batches
    total_error = running_error/num_batches
    elapsed = (time.time()-start)/60
    

    print('epoch=',epoch, '\t time=', elapsed,'min','\t lr=', my_lr  ,'\t loss=', total_loss , '\t error=', total_error*100 ,'percent')
    eval_on_test_set() 
    print(' ')
    
           


# ### Choose image at random from the test set and see how good/bad are the predictions

# In[ ]:


# choose a picture at random
idx=randint(0, 10000-1)
im=test_data[idx]

# diplay the picture
utils.show(im)

# send to device, rescale, and view as a batch of 1 
im = im.to(device)
im= (im-mean) / std
im=im.view(1,3,32,32)

# feed it to the net and display the confidence scores
scores =  net(im) 
probs= F.softmax(scores, dim=1)
utils.show_prob_cifar(probs.cpu())

