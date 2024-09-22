import torch
import torch.nn as nn
from torch.nn import init

class res_block(nn.Module):
    def __init__(self, in_ch):
        super(res_block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1)
        )
        self.nonlinear = nn.ReLU(inplace=True)
    def forward(self, x):
        res=self.conv1(x)
        output=res+x
        output=self.nonlinear(output)

        return output
    
class FusionNet(nn.Module):
    def __init__(self, config):
        super(FusionNet, self).__init__()
        feature_num = config['FusionNet']['feature_num']
        
        self.inc = nn.Sequential(
            nn.Conv2d(3, feature_num, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            res_block(feature_num),
            res_block(feature_num),
            res_block(feature_num),
            res_block(feature_num),
            res_block(feature_num)
        )

        self.inc2 = nn.Sequential(
            nn.Conv2d(3, feature_num, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            res_block(feature_num),
            res_block(feature_num),
            res_block(feature_num),
            res_block(feature_num),
            res_block(feature_num)
        )



        self.outc=nn.Sequential(
            nn.Conv2d(2*feature_num, feature_num, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            res_block(feature_num),
            res_block(feature_num),
            res_block(feature_num),
            nn.Conv2d(feature_num, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input1, input2):
        f1 = self.inc(input1)
        f2 = self.inc2(input2)
        con = torch.cat((f1,f2),dim =1)
        outf = self.outc(con)
        return outf



class RecNet(nn.Module):
    def __init__(self, config):
        super(RecNet, self).__init__()
        feature_num = config['FusionNet']['feature_num']
        self.inc = nn.Sequential(
            nn.Conv2d(9, feature_num, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            res_block(feature_num),
            res_block(feature_num),
            res_block(feature_num),
            res_block(feature_num),
            res_block(feature_num),
            res_block(feature_num),
            res_block(feature_num),
            res_block(feature_num),
            nn.Conv2d(feature_num, 6, kernel_size=3, stride=1, padding=1)
        )


    def forward(self, input1, input2, input3):
        in1 = torch.cat([input1,input2,input3], dim=1)
        outf = self.inc(in1)
        recA = outf[:,0:3,:,:]
        recB = outf[:,3:6,:,:]

        return recA, recB