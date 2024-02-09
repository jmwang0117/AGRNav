import numpy as np
import torch
from torch import nn
from torch.nn import init



class MobileViTv2Attention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, height, width):
        '''
        :param d_model: Output dimensionality of the model, which should match the channels number of the input.
        :param height: The height of the spatial dimension of the input.
        :param width: The width of the spatial dimension of the input.
        '''
        super(MobileViTv2Attention, self).__init__()
        self.fc_i = nn.Linear(height*width, 1)
        self.fc_k = nn.Linear(height*width, d_model)
        self.fc_v = nn.Linear(height*width, d_model)
        self.fc_o = nn.Linear(d_model, height*width)

        self.d_model = d_model
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :return:
        '''
        
        bs, ch, height, width = input.size()
        input = input.view(bs*ch, height*width).unsqueeze(-2)  # input shape becomes [bs*ch, 1, height*width]

        i = self.fc_i(input) #(bs*ch, 1, height*width)
        weight_i = torch.softmax(i, dim=-1) #bs*ch, 1, height*width
        context_score = weight_i * self.fc_k(input) #bs*ch, 1, height*width
        context_vector = torch.sum(context_score,dim=-2,keepdim=True) #bs*ch, 1, height*width
        v = self.fc_v(input) * context_vector #bs*ch, 1, height*width
        out = self.fc_o(v) #bs*ch, 1, height*width

        out = out.view(bs, ch, height, width) # reshape it to original input shape
        
        return out


# if __name__ == '__main__':
#     input=torch.randn(50,49,512)
#     sa = MobileViTv2Attention(d_model=512)
#     output=sa(input)
#     print(output.shape)

    