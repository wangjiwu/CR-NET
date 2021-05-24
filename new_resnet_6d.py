import torch.nn as nn
from model.transformer import Transformer
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']
from config import useHead, without_closs, use_transform, fix_step1_weight


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def tensor_split(t):
    arr = torch.split(t, 1, dim=4) # n*c*d*h*w
    #logger.info(arr[0].size())
    arr = [x.view(x.size()[:-1]) for x in arr]
    return arr

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv3x3_audio(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1,3), stride=(1, stride),
                     padding=(0,1), bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv1x1_audio(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1,1), stride=(1, stride), bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out +  identity
        out = self.relu(out)

        return out
class MLP(nn.Module):
    def __init__(self , in_size, output_dim,hidden = 100, dropout=0.3):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(in_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output_dim)
        self.dp = dropout

    def forward(self, din):
        dout = F.relu(self.fc1(din))
        dout = F.relu(self.fc2(dout))
        dout = F.dropout(self.fc3(dout), p=self.dp)
        return dout

class BasicBlock_audio(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_audio, self).__init__()
        self.conv1 = conv3x3_audio(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_audio(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out +  identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class Bottleneck_audio(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_audio, self).__init__()
        self.conv1 = conv1x1_audio(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1_audio(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out +  identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_output=4, zero_init_residual=False, sn=32):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool_all = nn.AdaptiveAvgPool2d((1, 1))

        self.avgpool_e = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_n = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_a = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_c = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_o = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_i = nn.AdaptiveAvgPool2d((1, 1))

        self.oneconv = nn.Conv1d(sn, 1, 1)

        self.featconv_e = nn.Conv2d(1, 4, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
        self.featconv_n = nn.Conv2d(1, 4, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
        self.featconv_a = nn.Conv2d(1, 4, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
        self.featconv_c = nn.Conv2d(1, 4, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
        self.featconv_o = nn.Conv2d(1, 4, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
        self.featconv_i = nn.Conv2d(1, 4, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)

        self.fc_e = nn.Linear(512 * block.expansion, num_output)
        self.fc_n = nn.Linear(512 * block.expansion, num_output)
        self.fc_a = nn.Linear(512 * block.expansion, num_output)
        self.fc_c = nn.Linear(512 * block.expansion, num_output)
        self.fc_o = nn.Linear(512 * block.expansion, num_output)
        self.fc_i = nn.Linear(512 * block.expansion, num_output)

        self.rfc_e = nn.Linear(512 * block.expansion, 1)
        self.rfc_n = nn.Linear(512 * block.expansion, 1)
        self.rfc_a = nn.Linear(512 * block.expansion, 1)
        self.rfc_c = nn.Linear(512 * block.expansion, 1)
        self.rfc_o = nn.Linear(512 * block.expansion, 1)
        self.rfc_i = nn.Linear(512 * block.expansion, 1)

        self.sofmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        def run_layer_on_arr(ar, l):
            return [l(t) for t in ar]

        #每一张image 都过一个Resnet34
        arr = tensor_split(x)
        arr = run_layer_on_arr(arr, self.conv1)
        arr = run_layer_on_arr(arr, self.bn1)
        arr = run_layer_on_arr(arr, self.relu)
        arr = run_layer_on_arr(arr, self.maxpool)

        arr = run_layer_on_arr(arr, self.layer1)
        arr = run_layer_on_arr(arr, self.layer2)
        arr = run_layer_on_arr(arr, self.layer3)
        arr = run_layer_on_arr(arr, self.layer4)



        arr_e = run_layer_on_arr(arr, self.avgpool_e)
        arr_n = run_layer_on_arr(arr, self.avgpool_n)
        arr_a = run_layer_on_arr(arr, self.avgpool_a)
        arr_c = run_layer_on_arr(arr, self.avgpool_c)
        arr_o = run_layer_on_arr(arr, self.avgpool_o)
        arr_i = run_layer_on_arr(arr, self.avgpool_i)

        arr_e = torch.cat([x.view(x.size(0), -1, 1) for x in arr_e], dim=2)
        arr_n = torch.cat([x.view(x.size(0), -1, 1) for x in arr_n], dim=2)
        arr_a = torch.cat([x.view(x.size(0), -1, 1) for x in arr_a], dim=2)
        arr_c = torch.cat([x.view(x.size(0), -1, 1) for x in arr_c], dim=2)
        arr_o = torch.cat([x.view(x.size(0), -1, 1) for x in arr_o], dim=2)
        arr_i = torch.cat([x.view(x.size(0), -1, 1) for x in arr_i], dim=2)

        # 32张图片通过一个 一维卷积进行特征合并  生成 一个512维向量
        # arr_e  bz * 32 * 512
        # 这里是 做了一个变化  是 把32 看成一个词向量的维度  所以 卷积核 大小为32 stride = 1  最终得到 512*32  -》 512
        re_e = self.oneconv(arr_e.permute(0, 2, 1))
        re_e = re_e.permute(0, 2, 1).view(re_e.size(0), -1)
        # 卷积操作  从 1个512维 变到  4 个512
        # re_e = bz * 512
        # regress_e  bz * 512 * 4
        regress_e = self.featconv_e(re_e.view(re_e.size(0), 1, -1, 1))

        re_n = self.oneconv(arr_n.permute(0, 2, 1))
        re_n = re_n.permute(0, 2, 1).view(re_n.size(0), -1)
        regress_n = self.featconv_n(re_n.view(re_n.size(0), 1, -1, 1))

        re_a = self.oneconv(arr_a.permute(0, 2, 1))
        re_a = re_a.permute(0, 2, 1).view(re_a.size(0), -1)
        regress_a = self.featconv_a(re_a.view(re_a.size(0), 1, -1, 1))

        re_c = self.oneconv(arr_c.permute(0, 2, 1))
        re_c = re_c.permute(0, 2, 1).view(re_c.size(0), -1)
        regress_c = self.featconv_c(re_c.view(re_c.size(0), 1, -1, 1))

        re_o = self.oneconv(arr_o.permute(0, 2, 1))
        re_o = re_o.permute(0, 2, 1).view(re_o.size(0), -1)
        regress_o = self.featconv_o(re_o.view(re_o.size(0), 1, -1, 1))

        re_i = self.oneconv(arr_i.permute(0, 2, 1))
        re_i = re_i.permute(0, 2, 1).view(re_i.size(0), -1)
        regress_i = self.featconv_i(re_i.view(re_i.size(0), 1, -1, 1))

        # 从512维向量算得权重  经过的是一个全连接 + sofmax 得到权重
        # w_e = bz * 4
        w_e = self.sofmax(self.fc_e(re_e)).view(re_e.size(0), -1, 1).expand(re_e.size(0), -1, 512)
        # 得到最终 加权后的 512维向量
        regress_e = torch.sum(torch.mul(w_e.view_as(regress_e), regress_e), 1, keepdim=True)

        w_n = self.sofmax(self.fc_n(re_n)).view(re_n.size(0), -1, 1).expand(re_n.size(0), -1, 512)
        regress_n = torch.sum(torch.mul(w_n.view_as(regress_n), regress_n), 1, keepdim=True)


        w_a = self.sofmax(self.fc_a(re_a)).view(re_a.size(0), -1, 1).expand(re_a.size(0), -1, 512)
        regress_a = torch.sum(torch.mul(w_a.view_as(regress_a), regress_a), 1, keepdim=True)


        w_c = self.sofmax(self.fc_c(re_c)).view(re_c.size(0), -1, 1).expand(re_c.size(0), -1, 512)
        regress_c = torch.sum(torch.mul(w_c.view_as(regress_c), regress_c), 1, keepdim=True)


        w_o = self.sofmax(self.fc_o(re_o)).view(re_o.size(0), -1, 1).expand(re_o.size(0), -1, 512)
        regress_o = torch.sum(torch.mul(w_o.view_as(regress_o), regress_o), 1, keepdim=True)


        w_i = self.sofmax(self.fc_i(re_i)).view(re_i.size(0), -1, 1).expand(re_i.size(0), -1, 512)
        regress_i = torch.sum(torch.mul(w_i.view_as(regress_i), regress_i), 1, keepdim=True)

        x_cls = [self.fc_e(re_e), self.fc_n(re_n), self.fc_a(re_a), self.fc_c(re_c), self.fc_o(re_o), self.fc_i(re_i)]
        # x_cls = [self.sofmax(self.fc_e(re_e)),
        #          self.sofmax(self.fc_n(re_n)),
        #          self.sofmax(self.fc_a(re_a)),
        #          self.sofmax(self.fc_c(re_c)),
        #          self.sofmax(self.fc_o(re_o)),
        #          self.sofmax(self.fc_i(re_i))]

        if without_closs:
            x_reg = [self.sigmoid(self.rfc_e(re_e.reshape(re_e.size(0), -1))),
                     self.sigmoid(self.rfc_n(re_n.reshape(re_n.size(0), -1))),
                     self.sigmoid(self.rfc_a(re_a.reshape(re_a.size(0), -1))),
                     self.sigmoid(self.rfc_c(re_c.reshape(re_c.size(0), -1))),
                     self.sigmoid(self.rfc_o(re_o.reshape(re_o.size(0), -1))),
                     self.sigmoid(self.rfc_i(re_i.reshape(re_i.size(0), -1)))]

        else:
            # x_reg = [self.sigmoid(self.rfc_e(regress_e.reshape(regress_e.size(0), -1))),
            #          self.sigmoid(self.rfc_n(regress_n.reshape(regress_n.size(0), -1))),
            #          self.sigmoid(self.rfc_a(regress_a.reshape(regress_a.size(0), -1))),
            #          self.sigmoid(self.rfc_c(regress_c.reshape(regress_c.size(0), -1))),
            #          self.sigmoid(self.rfc_o(regress_o.reshape(regress_o.size(0), -1))),
            #          self.sigmoid(self.rfc_i(regress_i.reshape(regress_i.size(0), -1)))]

            x_reg = [(self.rfc_e(regress_e.reshape(regress_e.size(0), -1))),
                     (self.rfc_n(regress_n.reshape(regress_n.size(0), -1))),
                     (self.rfc_a(regress_a.reshape(regress_a.size(0), -1))),
                     (self.rfc_c(regress_c.reshape(regress_c.size(0), -1))),
                     (self.rfc_o(regress_o.reshape(regress_o.size(0), -1))),
                     (self.rfc_i(regress_i.reshape(regress_i.size(0), -1)))]


        if without_closs:
            x_regress_result = torch.stack([re_e.reshape(re_e.size(0), -1),
                                            re_n.reshape(re_n.size(0), -1),
                                            re_a.reshape(re_a.size(0), -1),
                                            re_c.reshape(re_c.size(0), -1),
                                            re_o.reshape(re_o.size(0), -1),
                                            re_i.reshape(re_i.size(0), -1)], dim=-1)
        else:
            x_regress_result = torch.stack([regress_e.reshape(regress_e.size(0), -1),
                                            regress_n.reshape(regress_n.size(0), -1),
                                            regress_a.reshape(regress_a.size(0), -1),
                                            regress_c.reshape(regress_c.size(0), -1),
                                            regress_o.reshape(regress_o.size(0), -1),
                                            regress_i.reshape(regress_i.size(0), -1)], dim=-1)


        return x_cls, x_reg,x_regress_result


class ResNet_old(nn.Module):

    def __init__(self, block, layers, num_output=4, zero_init_residual=False, sn=32):
        super(ResNet_old, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool_e = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_n = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_a = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_c = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_o = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_i = nn.AdaptiveAvgPool2d((1, 1))
        self.sofmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        # for p in self.parameters():
        #     p.requires_grad = False

        self.oneconv = nn.Conv1d(sn, 1, 1)


        self.fc_e = nn.Linear(512 * block.expansion, num_output)
        self.fc_n = nn.Linear(512 * block.expansion, num_output)
        self.fc_a = nn.Linear(512 * block.expansion, num_output)
        self.fc_c = nn.Linear(512 * block.expansion, num_output)
        self.fc_o = nn.Linear(512 * block.expansion, num_output)
        self.fc_i = nn.Linear(512 * block.expansion, num_output)

        if fix_step1_weight:
            for p in self.parameters():
                p.requires_grad = False

        self.featconv_e = nn.Conv2d(1, 4, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
        self.featconv_n = nn.Conv2d(1, 4, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
        self.featconv_a = nn.Conv2d(1, 4, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
        self.featconv_c = nn.Conv2d(1, 4, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
        self.featconv_o = nn.Conv2d(1, 4, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
        self.featconv_i = nn.Conv2d(1, 4, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)


        self.rfc_e = nn.Linear(512 * block.expansion, 1)
        self.rfc_n = nn.Linear(512 * block.expansion, 1)
        self.rfc_a = nn.Linear(512 * block.expansion, 1)
        self.rfc_c = nn.Linear(512 * block.expansion, 1)
        self.rfc_o = nn.Linear(512 * block.expansion, 1)
        self.rfc_i = nn.Linear(512 * block.expansion, 1)




        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        def run_layer_on_arr(ar, l):
            return [l(t) for t in ar]

        arr = tensor_split(x)
        arr = run_layer_on_arr(arr, self.conv1)
        arr = run_layer_on_arr(arr, self.bn1)
        arr = run_layer_on_arr(arr, self.relu)
        arr = run_layer_on_arr(arr, self.maxpool)

        arr = run_layer_on_arr(arr, self.layer1)
        arr = run_layer_on_arr(arr, self.layer2)
        arr = run_layer_on_arr(arr, self.layer3)
        arr = run_layer_on_arr(arr, self.layer4)

        arr_e = run_layer_on_arr(arr, self.avgpool_e)
        arr_n = run_layer_on_arr(arr, self.avgpool_n)
        arr_a = run_layer_on_arr(arr, self.avgpool_a)
        arr_c = run_layer_on_arr(arr, self.avgpool_c)
        arr_o = run_layer_on_arr(arr, self.avgpool_o)
        arr_i = run_layer_on_arr(arr, self.avgpool_i)

        arr_e = torch.cat([x.view(x.size(0), -1, 1) for x in arr_e], dim=2)
        arr_n = torch.cat([x.view(x.size(0), -1, 1) for x in arr_n], dim=2)
        arr_a = torch.cat([x.view(x.size(0), -1, 1) for x in arr_a], dim=2)
        arr_c = torch.cat([x.view(x.size(0), -1, 1) for x in arr_c], dim=2)
        arr_o = torch.cat([x.view(x.size(0), -1, 1) for x in arr_o], dim=2)
        arr_i = torch.cat([x.view(x.size(0), -1, 1) for x in arr_i], dim=2)

        re_e = self.oneconv(arr_e.permute(0, 2, 1))
        re_e = re_e.permute(0, 2, 1).view(re_e.size(0), -1)
        regress_e = self.featconv_e(re_e.view(re_e.size(0), 1, -1, 1))

        re_n = self.oneconv(arr_n.permute(0, 2, 1))
        re_n = re_n.permute(0, 2, 1).view(re_n.size(0), -1)
        regress_n = self.featconv_n(re_n.view(re_n.size(0), 1, -1, 1))

        re_a = self.oneconv(arr_a.permute(0, 2, 1))
        re_a = re_a.permute(0, 2, 1).view(re_a.size(0), -1)
        regress_a = self.featconv_a(re_a.view(re_a.size(0), 1, -1, 1))

        re_c = self.oneconv(arr_c.permute(0, 2, 1))
        re_c = re_c.permute(0, 2, 1).view(re_c.size(0), -1)
        regress_c = self.featconv_c(re_c.view(re_c.size(0), 1, -1, 1))

        re_o = self.oneconv(arr_o.permute(0, 2, 1))
        re_o = re_o.permute(0, 2, 1).view(re_o.size(0), -1)
        regress_o = self.featconv_o(re_o.view(re_o.size(0), 1, -1, 1))

        re_i = self.oneconv(arr_i.permute(0, 2, 1))
        re_i = re_i.permute(0, 2, 1).view(re_i.size(0), -1)
        regress_i = self.featconv_i(re_i.view(re_i.size(0), 1, -1, 1))

        w_e = self.sofmax(self.fc_e(re_e)).view(re_e.size(0), -1, 1).expand(re_e.size(0), -1, 512)
        regress_e = torch.sum(torch.mul(w_e.view_as(regress_e), regress_e), 1, keepdim=True)

        w_n = self.sofmax(self.fc_n(re_n)).view(re_n.size(0), -1, 1).expand(re_n.size(0), -1, 512)
        regress_n = torch.sum(torch.mul(w_n.view_as(regress_n), regress_n), 1, keepdim=True)

        w_a = self.sofmax(self.fc_a(re_a)).view(re_a.size(0), -1, 1).expand(re_a.size(0), -1, 512)
        regress_a = torch.sum(torch.mul(w_a.view_as(regress_a), regress_a), 1, keepdim=True)

        w_c = self.sofmax(self.fc_c(re_c)).view(re_c.size(0), -1, 1).expand(re_c.size(0), -1, 512)
        regress_c = torch.sum(torch.mul(w_c.view_as(regress_c), regress_c), 1, keepdim=True)

        w_o = self.sofmax(self.fc_o(re_o)).view(re_o.size(0), -1, 1).expand(re_o.size(0), -1, 512)
        regress_o = torch.sum(torch.mul(w_o.view_as(regress_o), regress_o), 1, keepdim=True)

        w_i = self.sofmax(self.fc_i(re_i)).view(re_i.size(0), -1, 1).expand(re_i.size(0), -1, 512)
        regress_i = torch.sum(torch.mul(w_i.view_as(regress_i), regress_i), 1, keepdim=True)

        #x_cls = [self.fc_e(re_e), self.fc_n(re_n), self.fc_a(re_a), self.fc_c(re_c), self.fc_o(re_o), self.fc_i(re_i)]

        x_cls = [self.sofmax(self.fc_e(re_e)),
                 self.sofmax(self.fc_n(re_n)),
                 self.sofmax(self.fc_a(re_a)),
                 self.sofmax(self.fc_c(re_c)),
                 self.sofmax(self.fc_o(re_o)),
                 self.sofmax(self.fc_i(re_i))]


        x_reg = [self.rfc_e(regress_e.reshape(regress_e.size(0), -1)),
                 self.rfc_n(regress_n.reshape(regress_n.size(0), -1)),
                 self.rfc_a(regress_a.reshape(regress_a.size(0), -1)),
                 self.rfc_c(regress_c.reshape(regress_c.size(0), -1)),
                 self.rfc_o(regress_o.reshape(regress_o.size(0), -1)),
                 self.rfc_i(regress_i.reshape(regress_i.size(0), -1)), ]

        # logger.info(regress_e.reshape(regress_e.size(0), -1).size())

        ''''''
        if without_closs:
            x_regress_result = torch.stack([re_e.reshape(re_e.size(0), -1),
                                            re_n.reshape(re_n.size(0), -1),
                                            re_a.reshape(re_a.size(0), -1),
                                            re_c.reshape(re_c.size(0), -1),
                                            re_o.reshape(re_o.size(0), -1),
                                            re_i.reshape(re_i.size(0), -1)], dim=-1)
        else:
            x_regress_result = torch.stack([regress_e.reshape(regress_e.size(0), -1),
                                            regress_n.reshape(regress_n.size(0), -1),
                                            regress_a.reshape(regress_a.size(0), -1),
                                            regress_c.reshape(regress_c.size(0), -1),
                                            regress_o.reshape(regress_o.size(0), -1),
                                            regress_i.reshape(regress_i.size(0), -1)], dim=-1)

        return x_cls, x_reg, x_regress_result

class ResNet_old_classify(nn.Module):

    def __init__(self, block, layers, num_output=4, zero_init_residual=False, sn=32):
        super(ResNet_old_classify, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool_e = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_n = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_a = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_c = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_o = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_i = nn.AdaptiveAvgPool2d((1, 1))

        self.oneconv = nn.Conv1d(sn, 1, 1)


        self.fc_e = nn.Linear(512 * block.expansion, num_output)
        self.fc_n = nn.Linear(512 * block.expansion, num_output)
        self.fc_a = nn.Linear(512 * block.expansion, num_output)
        self.fc_c = nn.Linear(512 * block.expansion, num_output)
        self.fc_o = nn.Linear(512 * block.expansion, num_output)
        self.fc_i = nn.Linear(512 * block.expansion, num_output)

        self.sofmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        def run_layer_on_arr(ar, l):
            return [l(t) for t in ar]

        arr = tensor_split(x)
        arr = run_layer_on_arr(arr, self.conv1)
        arr = run_layer_on_arr(arr, self.bn1)
        arr = run_layer_on_arr(arr, self.relu)
        arr = run_layer_on_arr(arr, self.maxpool)

        arr = run_layer_on_arr(arr, self.layer1)
        arr = run_layer_on_arr(arr, self.layer2)
        arr = run_layer_on_arr(arr, self.layer3)
        arr = run_layer_on_arr(arr, self.layer4)

        arr_e = run_layer_on_arr(arr, self.avgpool_e)
        arr_n = run_layer_on_arr(arr, self.avgpool_n)
        arr_a = run_layer_on_arr(arr, self.avgpool_a)
        arr_c = run_layer_on_arr(arr, self.avgpool_c)
        arr_o = run_layer_on_arr(arr, self.avgpool_o)
        arr_i = run_layer_on_arr(arr, self.avgpool_i)

        arr_e = torch.cat([x.view(x.size(0), -1, 1) for x in arr_e], dim=2)
        arr_n = torch.cat([x.view(x.size(0), -1, 1) for x in arr_n], dim=2)
        arr_a = torch.cat([x.view(x.size(0), -1, 1) for x in arr_a], dim=2)
        arr_c = torch.cat([x.view(x.size(0), -1, 1) for x in arr_c], dim=2)
        arr_o = torch.cat([x.view(x.size(0), -1, 1) for x in arr_o], dim=2)
        arr_i = torch.cat([x.view(x.size(0), -1, 1) for x in arr_i], dim=2)

        re_e = self.oneconv(arr_e.permute(0, 2, 1))
        re_e = re_e.permute(0, 2, 1).view(re_e.size(0), -1)

        re_n = self.oneconv(arr_n.permute(0, 2, 1))
        re_n = re_n.permute(0, 2, 1).view(re_n.size(0), -1)

        re_a = self.oneconv(arr_a.permute(0, 2, 1))
        re_a = re_a.permute(0, 2, 1).view(re_a.size(0), -1)

        re_c = self.oneconv(arr_c.permute(0, 2, 1))
        re_c = re_c.permute(0, 2, 1).view(re_c.size(0), -1)

        re_o = self.oneconv(arr_o.permute(0, 2, 1))
        re_o = re_o.permute(0, 2, 1).view(re_o.size(0), -1)

        re_i = self.oneconv(arr_i.permute(0, 2, 1))
        re_i = re_i.permute(0, 2, 1).view(re_i.size(0), -1)


        x_cls = [self.fc_e(re_e), self.fc_n(re_n), self.fc_a(re_a), self.fc_c(re_c), self.fc_o(re_o), self.fc_i(re_i)]

        # x_cls = [self.sofmax(self.fc_e(re_e)),
        #          self.sofmax(self.fc_n(re_n)),
        #          self.sofmax(self.fc_a(re_a)),
        #          self.sofmax(self.fc_c(re_c)),
        #          self.sofmax(self.fc_o(re_o)),
        #          self.sofmax(self.fc_i(re_i))]
        # x_cls = [self.sofmax(self.sofmax(self.fc_e(re_e))),
        #          self.sofmax(self.sofmax(self.fc_n(re_n))),
        #          self.sofmax(self.sofmax(self.fc_a(re_a))),
        #          self.sofmax(self.sofmax(self.fc_c(re_c))),
        #          self.sofmax(self.sofmax(self.fc_o(re_o))),
        #          self.sofmax(self.sofmax(self.fc_i(re_i))]

        return x_cls


class ResNet_old_pretrain(nn.Module):

    def __init__(self, num_output=4, zero_init_residual=False, sn=32):
        super(ResNet_old_pretrain, self).__init__()


        self.oneconv = nn.Conv1d(sn, 1, 1)

        self.featconv_e = nn.Conv2d(1, 4, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
        self.featconv_n = nn.Conv2d(1, 4, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
        self.featconv_a = nn.Conv2d(1, 4, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
        self.featconv_c = nn.Conv2d(1, 4, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
        self.featconv_o = nn.Conv2d(1, 4, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
        self.featconv_i = nn.Conv2d(1, 4, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)

        self.fc_e = nn.Linear(512, num_output)
        self.fc_n = nn.Linear(512, num_output)
        self.fc_a = nn.Linear(512, num_output)
        self.fc_c = nn.Linear(512, num_output)
        self.fc_o = nn.Linear(512, num_output)
        self.fc_i = nn.Linear(512, num_output)

        self.rfc_e = nn.Linear(512, 1)
        self.rfc_n = nn.Linear(512, 1)
        self.rfc_a = nn.Linear(512, 1)
        self.rfc_c = nn.Linear(512, 1)
        self.rfc_o = nn.Linear(512, 1)
        self.rfc_i = nn.Linear(512, 1)


        self.sofmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()



    def forward(self, x):

        arr_e = arr_n = arr_a = arr_c = arr_o = arr_i = x.squeeze(-1).squeeze(-1).permute(0, 2, 1)



        re_e = self.oneconv(arr_e.permute(0, 2, 1))
        re_e = re_e.permute(0, 2, 1).view(re_e.size(0), -1)
        regress_e = self.featconv_e(re_e.view(re_e.size(0), 1, -1, 1))

        re_n = self.oneconv(arr_n.permute(0, 2, 1))
        re_n = re_n.permute(0, 2, 1).view(re_n.size(0), -1)
        regress_n = self.featconv_n(re_n.view(re_n.size(0), 1, -1, 1))

        re_a = self.oneconv(arr_a.permute(0, 2, 1))
        re_a = re_a.permute(0, 2, 1).view(re_a.size(0), -1)
        regress_a = self.featconv_a(re_a.view(re_a.size(0), 1, -1, 1))

        re_c = self.oneconv(arr_c.permute(0, 2, 1))
        re_c = re_c.permute(0, 2, 1).view(re_c.size(0), -1)
        regress_c = self.featconv_c(re_c.view(re_c.size(0), 1, -1, 1))

        re_o = self.oneconv(arr_o.permute(0, 2, 1))
        re_o = re_o.permute(0, 2, 1).view(re_o.size(0), -1)
        regress_o = self.featconv_o(re_o.view(re_o.size(0), 1, -1, 1))

        re_i = self.oneconv(arr_i.permute(0, 2, 1))
        re_i = re_i.permute(0, 2, 1).view(re_i.size(0), -1)
        regress_i = self.featconv_i(re_i.view(re_i.size(0), 1, -1, 1))

        w_e = self.sofmax(self.fc_e(re_e)).view(re_e.size(0), -1, 1).expand(re_e.size(0), -1, 512)
        regress_e = torch.sum(torch.mul(w_e.view_as(regress_e), regress_e), 1, keepdim=True)

        w_n = self.sofmax(self.fc_n(re_n)).view(re_n.size(0), -1, 1).expand(re_n.size(0), -1, 512)
        regress_n = torch.sum(torch.mul(w_n.view_as(regress_n), regress_n), 1, keepdim=True)

        w_a = self.sofmax(self.fc_a(re_a)).view(re_a.size(0), -1, 1).expand(re_a.size(0), -1, 512)
        regress_a = torch.sum(torch.mul(w_a.view_as(regress_a), regress_a), 1, keepdim=True)

        w_c = self.sofmax(self.fc_c(re_c)).view(re_c.size(0), -1, 1).expand(re_c.size(0), -1, 512)
        regress_c = torch.sum(torch.mul(w_c.view_as(regress_c), regress_c), 1, keepdim=True)

        w_o = self.sofmax(self.fc_o(re_o)).view(re_o.size(0), -1, 1).expand(re_o.size(0), -1, 512)
        regress_o = torch.sum(torch.mul(w_o.view_as(regress_o), regress_o), 1, keepdim=True)

        w_i = self.sofmax(self.fc_i(re_i)).view(re_i.size(0), -1, 1).expand(re_i.size(0), -1, 512)
        regress_i = torch.sum(torch.mul(w_i.view_as(regress_i), regress_i), 1, keepdim=True)

        #x_cls = [self.fc_e(re_e), self.fc_n(re_n), self.fc_a(re_a), self.fc_c(re_c), self.fc_o(re_o), self.fc_i(re_i)]

        x_cls = [self.sofmax(self.fc_e(re_e)),
                 self.sofmax(self.fc_n(re_n)),
                 self.sofmax(self.fc_a(re_a)),
                 self.sofmax(self.fc_c(re_c)),
                 self.sofmax(self.fc_o(re_o)),
                 self.sofmax(self.fc_i(re_i))]


        x_reg = [self.rfc_e(regress_e.reshape(regress_e.size(0), -1)),
                 self.rfc_n(regress_n.reshape(regress_n.size(0), -1)),
                 self.rfc_a(regress_a.reshape(regress_a.size(0), -1)),
                 self.rfc_c(regress_c.reshape(regress_c.size(0), -1)),
                 self.rfc_o(regress_o.reshape(regress_o.size(0), -1)),
                 self.rfc_i(regress_i.reshape(regress_i.size(0), -1)), ]


        ''''''
        if without_closs:
            x_regress_result = torch.stack([re_e.reshape(re_e.size(0), -1),
                                            re_n.reshape(re_n.size(0), -1),
                                            re_a.reshape(re_a.size(0), -1),
                                            re_c.reshape(re_c.size(0), -1),
                                            re_o.reshape(re_o.size(0), -1),
                                            re_i.reshape(re_i.size(0), -1)], dim=-1)
        else:
            x_regress_result = torch.stack([regress_e.reshape(regress_e.size(0), -1),
                                            regress_n.reshape(regress_n.size(0), -1),
                                            regress_a.reshape(regress_a.size(0), -1),
                                            regress_c.reshape(regress_c.size(0), -1),
                                            regress_o.reshape(regress_o.size(0), -1),
                                            regress_i.reshape(regress_i.size(0), -1)], dim=-1)

        return x_cls, x_reg, x_regress_result


class ResNet_transformer(nn.Module):

    def __init__(self, block, layers, num_output=4, zero_init_residual=False, sn=32):
        super(ResNet_transformer, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.oneconv = nn.Conv1d(sn, 1, 1)

        self.avgpool_all = nn.AdaptiveAvgPool2d((1, 1))





        self.transformer = Transformer()

        self.maxpool_all = nn.MaxPool2d((sn,1))


        self.featconv_e = nn.Conv2d(1, 4, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
        self.featconv_n = nn.Conv2d(1, 4, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
        self.featconv_a = nn.Conv2d(1, 4, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
        self.featconv_c = nn.Conv2d(1, 4, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
        self.featconv_o = nn.Conv2d(1, 4, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
        self.featconv_i = nn.Conv2d(1, 4, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)

        self.fc_e = nn.Linear(512 * block.expansion, num_output)
        self.fc_n = nn.Linear(512 * block.expansion, num_output)
        self.fc_a = nn.Linear(512 * block.expansion, num_output)
        self.fc_c = nn.Linear(512 * block.expansion, num_output)
        self.fc_o = nn.Linear(512 * block.expansion, num_output)
        self.fc_i = nn.Linear(512 * block.expansion, num_output)

        self.rfc_e = nn.Linear(512 * block.expansion, 1)
        self.rfc_n = nn.Linear(512 * block.expansion, 1)
        self.rfc_a = nn.Linear(512 * block.expansion, 1)
        self.rfc_c = nn.Linear(512 * block.expansion, 1)
        self.rfc_o = nn.Linear(512 * block.expansion, 1)
        self.rfc_i = nn.Linear(512 * block.expansion, 1)

        self.sofmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):

        def run_layer_on_arr(ar, l):
            return [l(t) for t in ar]

        # 过 resnet
        arr = tensor_split(x)
        arr = run_layer_on_arr(arr, self.conv1)
        arr = run_layer_on_arr(arr, self.bn1)
        arr = run_layer_on_arr(arr, self.relu)
        arr = run_layer_on_arr(arr, self.maxpool)

        arr = run_layer_on_arr(arr, self.layer1)
        arr = run_layer_on_arr(arr, self.layer2)
        arr = run_layer_on_arr(arr, self.layer3)
        arr = run_layer_on_arr(arr, self.layer4)

        arr = run_layer_on_arr(arr, self.avgpool_all)

        arr = torch.stack(arr, dim=1).squeeze(-1).squeeze(-1)

        # input bz * sn * 512
        enc_outputs, enc_self_attns = self.transformer(arr)
        # output bz * sn * 512

        # arr_e bz * 512
        enc_outputs = enc_outputs.unsqueeze(1).permute(2, 1, 0, 3)
        #arr = self.maxpool_all(enc_outputs)

        #arr = arr.squeeze(1)

        re_e= re_n=re_a= re_c= re_i= re_o = self.oneconv(enc_outputs.squeeze(1)).squeeze(1)
        # re_n = self.oneconv(enc_outputs.squeeze(1))


        regress_e = self.featconv_e(re_e.view(re_e.size(0), 1, -1, 1))

        regress_n = self.featconv_n(re_n.view(re_n.size(0), 1, -1, 1))

        regress_a = self.featconv_a(re_a.view(re_a.size(0), 1, -1, 1))

        regress_c = self.featconv_c(re_c.view(re_c.size(0), 1, -1, 1))

        regress_o = self.featconv_o(re_o.view(re_o.size(0), 1, -1, 1))

        regress_i = self.featconv_i(re_i.view(re_i.size(0), 1, -1, 1))

        w_e = self.sofmax(self.fc_e(re_e)).view(re_e.size(0), -1, 1).expand(re_e.size(0), -1, 512)
        regress_e = torch.sum(torch.mul(w_e.view_as(regress_e), regress_e), 1, keepdim=True)

        w_n = self.sofmax(self.fc_n(re_n)).view(re_n.size(0), -1, 1).expand(re_n.size(0), -1, 512)
        regress_n = torch.sum(torch.mul(w_n.view_as(regress_n), regress_n), 1, keepdim=True)

        w_a = self.sofmax(self.fc_a(re_a)).view(re_a.size(0), -1, 1).expand(re_a.size(0), -1, 512)
        regress_a = torch.sum(torch.mul(w_a.view_as(regress_a), regress_a), 1, keepdim=True)

        w_c = self.sofmax(self.fc_c(re_c)).view(re_c.size(0), -1, 1).expand(re_c.size(0), -1, 512)
        regress_c = torch.sum(torch.mul(w_c.view_as(regress_c), regress_c), 1, keepdim=True)

        w_o = self.sofmax(self.fc_o(re_o)).view(re_o.size(0), -1, 1).expand(re_o.size(0), -1, 512)
        regress_o = torch.sum(torch.mul(w_o.view_as(regress_o), regress_o), 1, keepdim=True)

        w_i = self.sofmax(self.fc_i(re_i)).view(re_i.size(0), -1, 1).expand(re_i.size(0), -1, 512)
        regress_i = torch.sum(torch.mul(w_i.view_as(regress_i), regress_i), 1, keepdim=True)

        x_cls = [self.fc_e(re_e), self.fc_n(re_n), self.fc_a(re_a), self.fc_c(re_c), self.fc_o(re_o), self.fc_i(re_i)]

        x_reg = [self.sigmoid(self.rfc_e(regress_e.reshape(regress_e.size(0), -1))),
                 self.rfc_n(regress_n.reshape(regress_n.size(0), -1)),
                 self.sigmoid(self.rfc_a(regress_a.reshape(regress_a.size(0), -1))),
                 self.rfc_c(regress_c.reshape(regress_c.size(0), -1)),
                 self.sigmoid(self.rfc_o(regress_o.reshape(regress_o.size(0), -1))),
                 self.sigmoid(self.rfc_i(regress_i.reshape(regress_i.size(0), -1))), ]

        # logger.info(regress_e.reshape(regress_e.size(0), -1).size())

        ''''''
        if without_closs:
            x_regress_result = torch.stack([re_e.reshape(re_e.size(0), -1),
                                            re_n.reshape(re_n.size(0), -1),
                                            re_a.reshape(re_a.size(0), -1),
                                            re_c.reshape(re_c.size(0), -1),
                                            re_o.reshape(re_o.size(0), -1),
                                            re_i.reshape(re_i.size(0), -1)], dim=-1)
        else:
            x_regress_result = torch.stack([regress_e.reshape(regress_e.size(0), -1),
                                            regress_n.reshape(regress_n.size(0), -1),
                                            regress_a.reshape(regress_a.size(0), -1),
                                            regress_c.reshape(regress_c.size(0), -1),
                                            regress_o.reshape(regress_o.size(0), -1),
                                            regress_i.reshape(regress_i.size(0), -1)], dim=-1)

        return x_cls, x_reg, x_regress_result

class ResNet_single(nn.Module):

    def __init__(self, block, layers, num_output=4, zero_init_residual=False, sn=32):
        super(ResNet_single, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.squeeze(0).permute(3,0,1,2)


        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)




        return x

class ResNet_audio(nn.Module):

    def __init__(self, block, layers, num_output=4, zero_init_residual=False, sn=1):
        super(ResNet_audio, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1,7), stride=(1,2), padding=(0,1),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2), padding=(0,1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool_e = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_n = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_a = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_c = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_o = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_i = nn.AdaptiveAvgPool2d((1, 1))

        self.oneconv = nn.Conv1d(sn, 1, 1)

        self.featconv_e = nn.Conv2d(1, 4, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False)
        self.featconv_n = nn.Conv2d(1, 4, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False)
        self.featconv_a = nn.Conv2d(1, 4, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False)
        self.featconv_c = nn.Conv2d(1, 4, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False)
        self.featconv_o = nn.Conv2d(1, 4, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False)
        self.featconv_i = nn.Conv2d(1, 4, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False)

        self.fc_e = nn.Linear(512 * block.expansion, num_output)
        self.fc_n = nn.Linear(512 * block.expansion, num_output)
        self.fc_a = nn.Linear(512 * block.expansion, num_output)
        self.fc_c = nn.Linear(512 * block.expansion, num_output)
        self.fc_o = nn.Linear(512 * block.expansion, num_output)
        self.fc_i = nn.Linear(512 * block.expansion, num_output)

        self.rfc_e = nn.Linear(512 * block.expansion, 1)
        self.rfc_n = nn.Linear(512 * block.expansion, 1)
        self.rfc_a = nn.Linear(512 * block.expansion, 1)
        self.rfc_c = nn.Linear(512 * block.expansion, 1)
        self.rfc_o = nn.Linear(512 * block.expansion, 1)
        self.rfc_i = nn.Linear(512 * block.expansion, 1)

        self.sofmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck_audio):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock_audio):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1_audio(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        bz, len = x.shape


        # 每一张image 都过一个Resnet34
        x = x.view(bz,1,1,len)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        arr_e = self.avgpool_e(x)
        arr_n = self.avgpool_n(x)
        arr_a = self.avgpool_a(x)
        arr_c = self.avgpool_c(x)
        arr_o = self.avgpool_o(x)
        arr_i = self.avgpool_i(x)

        arr_e = torch.cat([x.view(x.size(0), -1, 1) for x in arr_e], dim=2)
        arr_n = torch.cat([x.view(x.size(0), -1, 1) for x in arr_n], dim=2)
        arr_a = torch.cat([x.view(x.size(0), -1, 1) for x in arr_a], dim=2)
        arr_c = torch.cat([x.view(x.size(0), -1, 1) for x in arr_c], dim=2)
        arr_o = torch.cat([x.view(x.size(0), -1, 1) for x in arr_o], dim=2)
        arr_i = torch.cat([x.view(x.size(0), -1, 1) for x in arr_i], dim=2)

        # 32张图片通过一个 一维卷积进行特征合并  生成 一个512维向量
        # arr_e  bz * 32 * 512
        # 这里是 做了一个变化  是 把32 看成一个词向量的维度  所以 卷积核 大小为32 stride = 1  最终得到 512*32  -》 512
        re_e = self.oneconv(arr_e.permute(2, 1, 0))
        re_e = re_e.permute(0, 2, 1).view(re_e.size(0), -1)
        # 卷积操作  从 1个512维 变到  4 个512
        # re_e = bz * 512
        # regress_e  bz * 512 * 4
        regress_e = self.featconv_e(re_e.view(re_e.size(0), 1, -1, 1))

        re_n = self.oneconv(arr_n.permute(2, 1, 0))
        re_n = re_n.permute(0, 2, 1).view(re_n.size(0), -1)
        regress_n = self.featconv_n(re_n.view(re_n.size(0), 1, -1, 1))

        re_a = self.oneconv(arr_a.permute(2, 1, 0))
        re_a = re_a.permute(0, 2, 1).view(re_a.size(0), -1)
        regress_a = self.featconv_a(re_a.view(re_a.size(0), 1, -1, 1))

        re_c = self.oneconv(arr_c.permute(2, 1, 0))
        re_c = re_c.permute(0, 2, 1).view(re_c.size(0), -1)
        regress_c = self.featconv_c(re_c.view(re_c.size(0), 1, -1, 1))

        re_o = self.oneconv(arr_o.permute(2, 1, 0))
        re_o = re_o.permute(0, 2, 1).view(re_o.size(0), -1)
        regress_o = self.featconv_o(re_o.view(re_o.size(0), 1, -1, 1))

        re_i = self.oneconv(arr_i.permute(2, 1, 0))
        re_i = re_i.permute(0, 2, 1).view(re_i.size(0), -1)
        regress_i = self.featconv_i(re_i.view(re_i.size(0), 1, -1, 1))

        # 从512维向量算得权重  经过的是一个全连接 + sofmax 得到权重
        # w_e = bz * 4
        w_e = self.sofmax(self.fc_e(re_e)).view(re_e.size(0), -1, 1).expand(re_e.size(0), -1, 512)
        # 得到最终 加权后的 512维向量
        regress_e = torch.sum(torch.mul(w_e.view_as(regress_e), regress_e), 1, keepdim=True)

        w_n = self.sofmax(self.fc_n(re_n)).view(re_n.size(0), -1, 1).expand(re_n.size(0), -1, 512)
        regress_n = torch.sum(torch.mul(w_n.view_as(regress_n), regress_n), 1, keepdim=True)

        w_a = self.sofmax(self.fc_a(re_a)).view(re_a.size(0), -1, 1).expand(re_a.size(0), -1, 512)
        regress_a = torch.sum(torch.mul(w_a.view_as(regress_a), regress_a), 1, keepdim=True)

        w_c = self.sofmax(self.fc_c(re_c)).view(re_c.size(0), -1, 1).expand(re_c.size(0), -1, 512)
        regress_c = torch.sum(torch.mul(w_c.view_as(regress_c), regress_c), 1, keepdim=True)

        w_o = self.sofmax(self.fc_o(re_o)).view(re_o.size(0), -1, 1).expand(re_o.size(0), -1, 512)
        regress_o = torch.sum(torch.mul(w_o.view_as(regress_o), regress_o), 1, keepdim=True)

        w_i = self.sofmax(self.fc_i(re_i)).view(re_i.size(0), -1, 1).expand(re_i.size(0), -1, 512)
        regress_i = torch.sum(torch.mul(w_i.view_as(regress_i), regress_i), 1, keepdim=True)

        # # 得到分类预测值

        x_cls = [self.fc_e(re_e),
                 self.fc_n(re_n),
                 self.fc_a(re_a),
                 self.fc_c(re_c),
                 self.fc_o(re_o),
                 self.fc_i(re_i)]

        x_reg = [self.rfc_e(regress_e.reshape(regress_e.size(0), -1)),
                 self.rfc_n(regress_n.reshape(regress_n.size(0), -1)),
                 self.rfc_a(regress_a.reshape(regress_a.size(0), -1)),
                 self.rfc_c(regress_c.reshape(regress_c.size(0), -1)),
                 self.rfc_o(regress_o.reshape(regress_o.size(0), -1)),
                 self.rfc_i(regress_i.reshape(regress_i.size(0), -1)), ]

        if without_closs:
            x_regress_result = torch.stack([re_e.reshape(re_e.size(0), -1),
                                            re_n.reshape(re_n.size(0), -1),
                                            re_a.reshape(re_a.size(0), -1),
                                            re_c.reshape(re_c.size(0), -1),
                                            re_o.reshape(re_o.size(0), -1),
                                            re_i.reshape(re_i.size(0), -1)], dim=-1)
        else:

            x_regress_result = torch.stack([regress_e.reshape(regress_e.size(0), -1),
                                            regress_n.reshape(regress_n.size(0), -1),
                                            regress_a.reshape(regress_a.size(0), -1),
                                            regress_c.reshape(regress_c.size(0), -1),
                                            regress_o.reshape(regress_o.size(0), -1),
                                            regress_i.reshape(regress_i.size(0), -1)], dim=-1)

        return x_cls, x_reg, x_regress_result

class ResNet_audio_classify(nn.Module):

    def __init__(self, block, layers, num_output=4, zero_init_residual=False, sn=1):
        super(ResNet_audio_classify, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1,7), stride=(1,2), padding=(0,1),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2), padding=(0,1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool_e = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_n = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_a = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_c = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_o = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_i = nn.AdaptiveAvgPool2d((1, 1))

        self.oneconv = nn.Conv1d(sn, 1, 1)


        self.fc_e = nn.Linear(512 * block.expansion, num_output)
        self.fc_n = nn.Linear(512 * block.expansion, num_output)
        self.fc_a = nn.Linear(512 * block.expansion, num_output)
        self.fc_c = nn.Linear(512 * block.expansion, num_output)
        self.fc_o = nn.Linear(512 * block.expansion, num_output)
        self.fc_i = nn.Linear(512 * block.expansion, num_output)


        self.sofmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck_audio):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock_audio):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1_audio(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        bz, len = x.shape


        # 每一张image 都过一个Resnet34
        x = x.view(bz,1,1,len)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        arr_e = self.avgpool_e(x)
        arr_n = self.avgpool_n(x)
        arr_a = self.avgpool_a(x)
        arr_c = self.avgpool_c(x)
        arr_o = self.avgpool_o(x)
        arr_i = self.avgpool_i(x)

        arr_e = torch.cat([x.view(x.size(0), -1, 1) for x in arr_e], dim=2)
        arr_n = torch.cat([x.view(x.size(0), -1, 1) for x in arr_n], dim=2)
        arr_a = torch.cat([x.view(x.size(0), -1, 1) for x in arr_a], dim=2)
        arr_c = torch.cat([x.view(x.size(0), -1, 1) for x in arr_c], dim=2)
        arr_o = torch.cat([x.view(x.size(0), -1, 1) for x in arr_o], dim=2)
        arr_i = torch.cat([x.view(x.size(0), -1, 1) for x in arr_i], dim=2)

        # 32张图片通过一个 一维卷积进行特征合并  生成 一个512维向量
        # arr_e  bz * 32 * 512
        # 这里是 做了一个变化  是 把32 看成一个词向量的维度  所以 卷积核 大小为32 stride = 1  最终得到 512*32  -》 512
        re_e = self.oneconv(arr_e.permute(2, 1, 0))
        re_e = re_e.permute(0, 2, 1).view(re_e.size(0), -1)
        # 卷积操作  从 1个512维 变到  4 个512
        # re_e = bz * 512
        # regress_e  bz * 512 * 4

        re_n = self.oneconv(arr_n.permute(2, 1, 0))
        re_n = re_n.permute(0, 2, 1).view(re_n.size(0), -1)

        re_a = self.oneconv(arr_a.permute(2, 1, 0))
        re_a = re_a.permute(0, 2, 1).view(re_a.size(0), -1)

        re_c = self.oneconv(arr_c.permute(2, 1, 0))
        re_c = re_c.permute(0, 2, 1).view(re_c.size(0), -1)

        re_o = self.oneconv(arr_o.permute(2, 1, 0))
        re_o = re_o.permute(0, 2, 1).view(re_o.size(0), -1)

        re_i = self.oneconv(arr_i.permute(2, 1, 0))
        re_i = re_i.permute(0, 2, 1).view(re_i.size(0), -1)


        # # 得到分类预测值

        x_cls = [self.fc_e(re_e),
                 self.fc_n(re_n),
                 self.fc_a(re_a),
                 self.fc_c(re_c),
                 self.fc_o(re_o),
                 self.fc_i(re_i)]


        return x_cls


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

def resnet34_pretrain(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_single(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), False)
    return model

def resnet34_old( pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    if (use_transform):
        model = ResNet_transformer(BasicBlock, [3, 4, 6, 3], **kwargs)
    else:
        model = ResNet_old(BasicBlock, [3, 4, 6, 3], **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), False)

    return model

def resnet34_old_classify(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    if (use_transform):
        model = ResNet_transformer(BasicBlock, [3, 4, 6, 3], **kwargs)
    else:
        model = ResNet_old_classify(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']),False)
    return model

def resnet34_audio(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_audio(BasicBlock_audio, [3, 4, 6, 3], **kwargs)

    return model

def resnet34_audio_classify(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_audio_classify(BasicBlock_audio, [3, 4, 6, 3], **kwargs)

    return model

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


class ConvolutionBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvolutionBlock, self).__init__()
        self.conv  = nn.Conv2d(in_channels, out_channels, kernel_size=(1,49), stride=(1,4), padding=(0,24))
        self.bn_conv  = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):

        out = self.conv(x)
        out = self.bn_conv(out)
        out = self.relu(out)
        return out

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.res_branch2a   = nn.Conv2d(in_channels, out_channels, kernel_size=(1,9), padding=(0,4))
        self.bn_branch2a   = nn.BatchNorm2d(out_channels)
        self.res_branch2b = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 9), padding=(0, 4))
        self.bn_branch2b = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):

        h = self.res_branch2a(x)
        h = self.bn_branch2a(h)
        h = self.relu(h)
        h = self.res_branch2b(h)
        h = self.bn_branch2b(h)
        h = x + h
        y = self.relu(h)
        return y

class ResidualBlockB(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResidualBlockB, self).__init__()
        self.res_branch1    = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride= (1,4) , padding=(0,0))
        self.bn_branch1    = nn.BatchNorm2d(out_channels)
        self.res_branch2a     = nn.Conv2d(in_channels, out_channels, kernel_size=(1,9), stride= (1,4) , padding=(0,4))
        self.bn_branch2a    = nn.BatchNorm2d(out_channels)
        self.res_branch2b = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 9), padding=(0, 4))
        self.bn_branch2b = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        temp = self.res_branch1(x)
        temp = self.bn_branch1(temp)
        h = self.res_branch2a(x)
        h = self.bn_branch2a(h)
        h = self.relu(h)
        h = self.res_branch2b(h)
        h = self.bn_branch2b(h)
        h = temp + h
        y = self.relu(h)
        return y


class ResNet18_audio(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResNet18_audio, self).__init__()
        conv1_relu = ConvolutionBlock(1, 32)
        res2a_relu = ResidualBlock(32, 32)
        res2b_relu = ResidualBlock(32, 32)
        res3a_relu = ResidualBlockB(32, 64)
        res3b_relu = ResidualBlock(64, 64)
        res4a_relu = ResidualBlockB(64, 128)
        res4b_relu = ResidualBlock(128, 128)
        res5a_relu = ResidualBlockB(128, 256)
        res5b_relu = ResidualBlock(256, 256)
        self.relu = nn.ReLU(inplace=True)
        self.max_pooling_2d = nn.MaxPool2d(kernel_size=(1,9), stride=(1,4), padding=(0,4))
        self.average_pooling_2d = nn.AvgPool2d(kernel_size=(1,9), stride=(1,4), padding=(0,4))


    def forward(self, x):
        h = self.conv1_relu(x)
        h = self.max_pooling_2d(h, (1, 9), (1, 4), (0, 4))
        h = self.res2a_relu(h)
        h = self.res2b_relu(h)
        h = self.res3a_relu(h)
        h = self.res3b_relu(h)
        h = self.res4a_relu(h)
        h = self.res4b_relu(h)
        h = self.res5a_relu(h)
        h = self.res5b_relu(h)
        y = self.average_pooling_2d(h)
        return y