import torch
import torch.nn as nn

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

class ResNet(nn.Module):

    def __init__(self):

        super(ResNet, self).__init__()

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

