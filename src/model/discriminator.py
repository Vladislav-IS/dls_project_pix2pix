# generator.py и discriminator.py содержат код генератора и дискриминатора соответственно.
# При написании нейросети я опирался на оригинальную статью (https://arxiv.org/pdf/1611.07004.pdf)
# и на это видео с Ютуба: https://youtu.be/SuddDSqGRzg. Из видео я взял идею создать отдельные классы
# слоёв нейросетей, на основе которых создаются классы самих сетей. Генератор представляет собой
# укороченный UNet, а дискриминатор подобен патч-дискриминатору из оригинальной статьи (разве что на
# его выходе получается тензор 2х2, а не 70х70) и принимает на вход конкатенированный тензор
# "сегментированное изображение/оригинальный фасад" или "сегментированное изображение/сгенерированный фасад".

import torch
import torch.nn as nn

class DiscrConv(nn.Module):
  def __init__(self, in_channels, out_channels, layer_type='input_hidden', kernel_size=4, stride=2, padding=1):
    super().__init__()
    self.layer_type = layer_type
    self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
						  stride=stride, padding=padding)
    self.innorm = nn.InstanceNorm2d(out_channels, affine=True)
    self.leaky = nn.LeakyReLU(0.2)
    self.sigmoid = nn.Sigmoid()

  def forward(self, inp):
    inp = self.conv(inp)
    if self.layer_type == 'input_hidden':
      inp = self.innorm(inp)
      inp = self.leaky(inp)
    else:
      inp = self.sigmoid(inp)
    return inp

class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.c0 = DiscrConv(6, 64)
    self.c1 = DiscrConv(64, 128)
    self.c2 = DiscrConv(128, 256)
    self.c3 = DiscrConv(256, 512)
    self.c4 = DiscrConv(512, 512)
    self.c5 = DiscrConv(512, 512)
    self.c6 = DiscrConv(512, 1, layer_type='output')

  def forward(self, x, y):
    inp = torch.cat((x, y), dim=1)
    inp = self.c0(inp)
    inp = self.c1(inp)
    inp = self.c2(inp)
    inp = self.c3(inp)
    inp = self.c4(inp)
    inp = self.c5(inp)
    inp = self.c6(inp)
    return inp