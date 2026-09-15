
# https://github.com/prakashchhipa/ASTrA/blob/main/StrategyNet.py#L161


import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
from maskrcnn_benchmark.modeling.backbone.resnet import Bottleneck
import torchvision 
from maskrcnn_benchmark.engine.utils import save_img
# from resnet import Bottleneck

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet_Strategy(nn.Module):
    def __init__(self, block, num_blocks,  args, noises=-1, no_noise_pred=1):
        super(ResNet_Strategy, self).__init__()

        resnet18 = torchvision.models.resnet18(pretrained=True)

        resnet18.layer4[0].conv2.stride = (1,1)
        # resnet18.layer4[0].downsample[0].stride = (1,1)
        resnet18 = nn.Sequential(*list(resnet18.children())[:-2])

        self.base = resnet18
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # self.bn1 = nn.BatchNorm1d(256)
        # self.classifier = nn.Linear(self.middle_dim2 , self.num_classes, bias=False)

        self.no_noise_pred = no_noise_pred
        self.saved_log_probs = []
        self.saved_rewards = []

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
        #                        stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        # self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        self.noise_output  = nn.ModuleList()
        for i in range(self.no_noise_pred):
            noise = nn.Linear(512*block.expansion, noises, bias=False) 
            nn.init.kaiming_uniform_(noise.weight)
            # trunc_normal_(noise.weight, std=.02)
            self.noise_output.append(noise)
        
        self.saved_log_probs = []
        self.rewards = []
        self.R1s = []
        self.R2s = []
        self.R3s = []

        # for name,p in self.named_parameters():
        #     print(f"{name:<30} : {p.mean().item():.4f}")


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # save_img(x, name="strategy.png", batch_dim=True, )
        out = self.base(x)
        
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        out = self.avgpool(out)
        # torch.Size([10, 512, 1, 1])
        
        out = out.view(out.size(0), -1)
        # torch.Size([10, 512])

        output = [ ]
        for noise_output in self.noise_output:
            output.append( noise_output( out ) ) 
            
        return  output


def get_args():
    import argparse
    parser = argparse.ArgumentParser('LAS_AT')
    ## search
    parser.add_argument('--epsilon_types', type=list, default=range(1, 11)) # 11 class classifier 
    parser.add_argument('--attack_iters_types', type=list, default=range(1, 10)) # 10 class classifier 
    parser.add_argument('--step_size_types', type=list, default=range(1, 5)) # 4 class classifier 
    arguments = parser.parse_args()
    return arguments


def ResNet18_Strategy(args, noises, no_noise_pred):
    return ResNet_Strategy(BasicBlock, [2, 2, 2, 2], args, noises, no_noise_pred)

def ResNet34_Strategy(args, noises, no_noise_pred):
    return ResNet_Strategy(BasicBlock, [3, 4, 6, 3],args, noises, no_noise_pred)


def ResNet50_Strategy(args, noises, no_noise_pred):
    return ResNet_Strategy(Bottleneck, [3, 4, 6, 3],args, noises, no_noise_pred)


def ResNet101_Strategy(args, noises, no_noise_pred):
    return ResNet_Strategy(Bottleneck, [3, 4, 23, 3],args, noises, no_noise_pred)


if __name__=="__main__":
    args = get_args()
    noises = ['fog', 'rain']
    model=ResNet18_Strategy(args, noises)
    print(model)
    # torch.Size([2, 3, 640, 1152])
    print(model(torch.rand([2,3,640,1152]))[0].shape)

    # out = resnet18 ( torch.rand([2,3,640,1152]) ) 

# cd ~/robustness_object_detection/GLIP/
# python maskrcnn_benchmark/modeling/backbone/strategy_model.py