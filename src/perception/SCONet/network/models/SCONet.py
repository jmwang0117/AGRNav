import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from network.models.MobileViTv2Attention import MobileViTv2Attention
from network.models.CrissCrossAttention import CrissCrossAttention

class SegmentationHead(nn.Module):
  '''
  3D Segmentation heads to retrieve semantic segmentation at each scale.
  Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
  '''
  def __init__(self, inplanes, planes, nbr_classes, dilations_conv_list):
    super().__init__()

    # First convolution
    self.conv0 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, stride=1)

    # ASPP Block
    self.conv_list = dilations_conv_list
    self.conv1 = nn.ModuleList(
      [nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
    self.bn1 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
    self.conv2 = nn.ModuleList(
      [nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
    self.bn2 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
    self.relu = nn.ReLU(inplace=True)

    # Convolution for output
    self.conv_classes = nn.Conv3d(planes, nbr_classes, kernel_size=3, padding=1, stride=1)

  def forward(self, x_in):

    # Dimension exapension
    x_in = x_in[:, None, :, :, :]

    # Convolution to go from inplanes to planes features...
    x_in = self.relu(self.conv0(x_in))

    y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
    for i in range(1, len(self.conv_list)):
      y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
    x_in = self.relu(y + x_in)  # modified

    x_in = self.conv_classes(x_in)

    return x_in

class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1, bias=False):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size,
                                        stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.pointwise_conv(out)
        return out

class EncoderBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, mid_channels, kernel_size, padding, stride)
        self.relu1 = nn.ReLU()
        self.conv2 = DepthwiseSeparableConv(mid_channels, out_channels, kernel_size, padding, stride)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return x

class SCONet(nn.Module):

  def __init__(self, class_num, input_dimensions, class_frequencies):
    '''
    SCONet architecture
    :param N: number of classes to be predicted 
    '''
    super().__init__()
    self.nbr_classes = class_num
    self.input_dimensions = input_dimensions  # Grid dimensions should be (W, H, D).. z or height being axis 1
    self.class_frequencies = class_frequencies
    f = self.input_dimensions[1]

    self.pool = nn.MaxPool2d(2)  # [F=2; S=2; P=0; D=1]

    self.Encoder_block1 = nn.Sequential(
            DepthwiseSeparableConv(f, f, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            DepthwiseSeparableConv(f, f, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )

    self.Encoder_block2 = nn.Sequential(
            nn.MaxPool2d(2),
            DepthwiseSeparableConv(f, int(f*1.5), kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            DepthwiseSeparableConv(int(f*1.5), int(f*1.5), kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )

    self.Encoder_block3 = nn.Sequential(
            nn.MaxPool2d(2),
            DepthwiseSeparableConv(int(f*1.5), int(f*2), kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            DepthwiseSeparableConv(int(f*2), int(f*2), kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )

    self.Encoder_block4 = nn.Sequential(
            nn.MaxPool2d(2),
            DepthwiseSeparableConv(int(f*2), int(f*2.5), kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            DepthwiseSeparableConv(int(f*2.5), int(f*2.5), kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )

    self.criss_cross_attention = CrissCrossAttention(int(f*2.5))
    self.Attention_block_1_8 = MobileViTv2Attention(4, 64, 64)
    self.Attention_block_1_4 = MobileViTv2Attention(8, 128, 128)
    
    # Treatment output 1:8
    self.conv_out_scale_1_8 = nn.Conv2d(int(f*2.5), int(f/8), kernel_size=3, padding=1, stride=1)
    self.deconv_1_8__1_2    = nn.ConvTranspose2d(int(f/8), int(f/8), kernel_size=4, padding=0, stride=4)
    self.deconv_1_8__1_1    = nn.ConvTranspose2d(int(f/8), int(f/8), kernel_size=8, padding=0, stride=8)

    # Treatment output 1:4
    self.deconv1_8          = nn.ConvTranspose2d(int(f/8), int(f/8), kernel_size=6, padding=2, stride=2)
    self.conv1_4            = nn.Conv2d(int(f*2) + int(f/8), int(f*2), kernel_size=3, padding=1, stride=1)
    self.conv_out_scale_1_4 = nn.Conv2d(int(f*2), int(f/4), kernel_size=3, padding=1, stride=1)
    self.deconv_1_4__1_1    = nn.ConvTranspose2d(int(f/4), int(f/4), kernel_size=4, padding=0, stride=4)

    # Treatment output 1:2
    self.deconv1_4          = nn.ConvTranspose2d(int(f/4), int(f/4), kernel_size=6, padding=2, stride=2)
    self.conv1_2            = nn.Conv2d(int(f*1.5) + int(f/4) + int(f/8), int(f*1.5), kernel_size=3, padding=1, stride=1)
    self.conv_out_scale_1_2 = nn.Conv2d(int(f*1.5), int(f/2), kernel_size=3, padding=1, stride=1)

    # Treatment output 1:1
    self.deconv1_2          = nn.ConvTranspose2d(int(f/2), int(f/2), kernel_size=6, padding=2, stride=2)
    self.conv1_1            = nn.Conv2d(int(f/8) + int(f/4) + int(f/2) + int(f), f, kernel_size=3, padding=1, stride=1)
    self.seg_head_1_1       = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])

  def forward(self, x):

    input = x['3D_OCCUPANCY']  # Input to SCONet model is 3D occupancy big scale (1:1) [bs, 1, W, H, D]
    input = torch.squeeze(input, dim=1).permute(0, 2, 1, 3)  # Reshaping to the right way for 2D convs [bs, H, W, D]

    # Encoder block
    _skip_1_1 = self.Encoder_block1(input)
    _skip_1_2 = self.Encoder_block2(_skip_1_1)
    _skip_1_4 = self.Encoder_block3(_skip_1_2)
    _skip_1_8 = self.Encoder_block4(_skip_1_4)
    _skip_1_8 = self.criss_cross_attention(_skip_1_8)

    # Out 1_8
    out_scale_1_8__2D = self.conv_out_scale_1_8(_skip_1_8)
    out = self.deconv1_8(out_scale_1_8__2D)
    out = self.Attention_block_1_8(out) 
       

    # Out 1_4       
    out = torch.cat((out, _skip_1_4), 1)
    out = F.relu(self.conv1_4(out))
    out_scale_1_4__2D = self.conv_out_scale_1_4(out)
    out = self.deconv1_4(out_scale_1_4__2D)
    

    # Out 1_2    
    out = self.Attention_block_1_4(out)
    out = torch.cat((out, _skip_1_2, self.deconv_1_8__1_2(out_scale_1_8__2D)), 1)
    out = F.relu(self.conv1_2(out))
    out_scale_1_2__2D = self.conv_out_scale_1_2(out)
    

    # Out 1_1
    out = self.deconv1_2(out_scale_1_2__2D)
    out = torch.cat((out, _skip_1_1, self.deconv_1_4__1_1(out_scale_1_4__2D), self.deconv_1_8__1_1(out_scale_1_8__2D)), 1)
    out_scale_1_1__2D = F.relu(self.conv1_1(out))
    out_scale_1_1__3D = self.seg_head_1_1(out_scale_1_1__2D)

    # Take back to [W, H, D] axis order
    out_scale_1_1__3D = out_scale_1_1__3D.permute(0, 1, 3, 2, 4)  # [bs, C, H, W, D] -> [bs, C, W, H, D]
    scores = {'pred_semantic_1_1': out_scale_1_1__3D}
    return scores

  def weights_initializer(self, m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, DepthwiseSeparableConv):
        # 初始化Depthwise卷积层
        nn.init.kaiming_uniform_(m.depthwise_conv.weight, nonlinearity='relu')
        if m.depthwise_conv.bias is not None:
            nn.init.zeros_(m.depthwise_conv.bias)
        # 初始化Pointwise卷积层
        nn.init.kaiming_uniform_(m.pointwise_conv.weight, nonlinearity='relu')
        if m.pointwise_conv.bias is not None:
            nn.init.zeros_(m.pointwise_conv.bias)

  def weights_init(self):
    self.apply(self.weights_initializer)

  def get_parameters(self):
    return self.parameters()

  def compute_loss(self, scores, data):
    '''
    :param: prediction: the predicted tensor, must be [BS, C, H, W, D]
    '''

    target = data['3D_LABEL']['1_1']
    device, dtype = target.device, target.dtype
    class_weights = self.get_class_weights().to(device=target.device, dtype=target.dtype)

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255, reduction='mean').to(device=device)

    loss_1_1 = criterion(scores['pred_semantic_1_1'], data['3D_LABEL']['1_1'].long())

    loss = {'total': loss_1_1, 'semantic_1_1': loss_1_1}

    return loss

  def get_class_weights(self):
    '''
    Cless weights being 1/log(fc)
    '''
    epsilon_w = 0.001  # eps to avoid zero division
    weights = torch.from_numpy(1 / np.log(self.class_frequencies + epsilon_w))

    return weights

  def get_target(self, data):
    '''
    Return the target to use for evaluation of the model
    '''
    return {'1_1': data['3D_LABEL']['1_1']}
    # return data['3D_LABEL']['1_1'] #.permute(0, 2, 1, 3)

  def get_scales(self):
    '''
    Return scales needed to train the model
    '''
    scales = ['1_1']
    return scales

  def get_validation_loss_keys(self):
    return ['total', 'semantic_1_1']

  def get_train_loss_keys(self):
    return ['total', 'semantic_1_1']