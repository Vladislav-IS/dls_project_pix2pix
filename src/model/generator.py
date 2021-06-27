# generator.py и discriminator.py содержат код генератора и дискриминатора соответственно.
# При написании нейросети я опирался на оригинальную статью (https://arxiv.org/pdf/1611.07004.pdf)
# и на это видео с Ютуба: https://youtu.be/SuddDSqGRzg. Из видео я взял идею создать отдельные классы
# слоёв нейросетей, на основе которых создаются классы самих сетей. Генератор представляет собой
# укороченный UNet, а дискриминатор подобен патч-дискриминатору из оригинальной статьи (разве что на
# его выходе получается тензор 2х2, а не 70х70) и принимает на вход конкатенированный тензор
# "сегментированное изображение/оригинальный фасад" или "сегментированное изображение/сгенерированный фасад".

import torch
import torch.nn as nn

class GenConv(nn.Module):
  def __init__(self, in_channels, out_channels, layer_type='input_hidden', dropflag=False, upsflag=False):
    super().__init__()
    self.dropflag = dropflag
    self.upsflag = upsflag
    self.layer_type = layer_type
    if self.upsflag:
      self.conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
										  kernel_size=3, stride=1, padding=1))
    else:
      self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
							kernel_size=4, stride=2, padding=1)
    self.innorm = nn.InstanceNorm2d(out_channels, affine=True)
    self.dropout = nn.Dropout()
    self.relu = nn.ReLU()
    self.tanh = nn.Tanh()

  def forward(self, inp):
    inp = self.conv(inp)
    if self.layer_type == 'input_hidden':
      inp = self.innorm(inp)
      if self.dropflag:
        inp = self.dropout(inp)
      inp = self.relu(inp)
    else:
      inp = self.tanh(inp)
    return inp

class Generator(nn.Module):
  def __init__(self):
    super().__init__()
    self.in_layer = GenConv(3, 64)
    self.enc0 = GenConv(64, 128)
    self.enc1 = GenConv(128, 256)
    self.enc2 = GenConv(256, 512)
    self.enc3 = GenConv(512, 512)
    self.enc4 = GenConv(512, 512)
    self.bottleneck = GenConv(512, 512)
    self.dec0 = GenConv(512, 512, dropflag=True, upsflag=True)
    self.dec1 = GenConv(1024, 512, dropflag=True, upsflag=True)
    self.dec2 = GenConv(1024, 256, upsflag=True)
    self.dec3 = GenConv(768, 128, upsflag=True)
    self.dec4 = GenConv(384, 64, upsflag=True)
    self.dec5 = GenConv(192, 64, upsflag=True)
    self.out_layer = GenConv(128, 3, layer_type='output', upsflag=True)

  def forward(self, x):
    inp = self.in_layer(x)
    e0 = self.enc0(inp)
    e1 = self.enc1(e0)
    e2 = self.enc2(e1)
    e3 = self.enc3(e2)
    e4 = self.enc4(e3)
    b = self.bottleneck(e4)
    d0 = self.dec0(b)
    d1 = self.dec1(torch.cat((e4, d0), dim=1))
    d2 = self.dec2(torch.cat((e3, d1), dim=1))
    d3 = self.dec3(torch.cat((e2, d2), dim=1))
    d4 = self.dec4(torch.cat((e1, d3), dim=1))
    d5 = self.dec5(torch.cat((e0, d4), dim=1))
    outp = self.out_layer(torch.cat((inp, d5), dim=1))
    return outp
