import torch
import torch.nn as nn


class DenseNet2D_down_block(nn.Module):
    def __init__(self,input_channels,output_channels,down_size,dropout=False,prob=0):
        super(DenseNet2D_down_block, self).__init__()
        self.conv1 = nn.Conv3d(input_channels,output_channels,kernel_size=(1,3,3),padding=(0,1,1))
        self.conv21 = nn.Conv3d(input_channels+output_channels,output_channels,kernel_size=(1,1,1),padding=(0,0,0))
        self.conv22 = nn.Conv3d(output_channels,output_channels,kernel_size=(1,3,3),padding=(0,1,1))
        self.conv31 = nn.Conv3d(input_channels+2*output_channels,output_channels,kernel_size=(1,1,1),padding=(0,0,0))
        self.conv32 = nn.Conv3d(output_channels,output_channels,kernel_size=(1,3,3),padding=(0,1,1))
        self.max_pool = nn.MaxPool3d(kernel_size=down_size)
        self.relu = nn.LeakyReLU()
        self.down_size = down_size
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)
        self.dropout3 = nn.Dropout(p=prob)

    def forward(self, x):
        if self.down_size != None:
            x = self.max_pool(x)
            
        if self.dropout:
            x1 = self.relu(self.dropout1(self.conv1(x)))
            x21 = torch.cat((x,x1),dim=1)
            x22 = self.relu(self.dropout2(self.conv22(self.conv21(x21))))
            x31 = torch.cat((x21,x22),dim=1)
            out = self.relu(self.dropout3(self.conv32(self.conv31(x31))))
        else:
            x1 = self.relu(self.conv1(x))
            x21 = torch.cat((x,x1),dim=1)
            x22 = self.relu(self.conv22(self.conv21(x21)))
            x31 = torch.cat((x21,x22),dim=1)
            out = self.relu(self.conv32(self.conv31(x31)))
        
        return out

    
class DenseNet2D_up_block_concat(nn.Module):
    def __init__(self,skip_channels,input_channels,output_channels,up_stride,dropout=False,prob=0):
        super(DenseNet2D_up_block_concat, self).__init__()
        self.conv11 = nn.Conv3d(skip_channels+input_channels,output_channels,kernel_size=(1,1,1),padding=(0,0,0))
        self.conv12 = nn.Conv3d(output_channels,output_channels,kernel_size=(1,3,3),padding=(0,1,1))
        self.conv21 = nn.Conv3d(skip_channels+input_channels+output_channels,output_channels,
                                kernel_size=(1,1,1),padding=(0,0,0))
        self.conv22 = nn.Conv3d(output_channels,output_channels,kernel_size=(1,3,3),padding=(0,1,1))
        self.relu = nn.LeakyReLU()
        self.up_stride = up_stride
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)

    def forward(self,prev_feature_map,x):
        x = nn.functional.interpolate(x,scale_factor=self.up_stride,mode='nearest')
        x = torch.cat((x,prev_feature_map),dim=1)
        if self.dropout:
            x1 = self.relu(self.dropout1(self.conv12(self.conv11(x))))
            x21 = torch.cat((x,x1),dim=1)
            out = self.relu(self.dropout2(self.conv22(self.conv21(x21))))
        else:
            x1 = self.relu(self.conv12(self.conv11(x)))
            x21 = torch.cat((x,x1),dim=1)
            out = self.relu(self.conv22(self.conv21(x21)))
            
        return out
