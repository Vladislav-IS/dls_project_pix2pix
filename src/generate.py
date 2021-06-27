# generate.py представляет собой полностью автономный файл с интерфейсом для генерации
# изображений (хотя ему все равно необходим файл model3_g.pth с моделью генератора).
# Из тех же соображений автономности классы внутри generate.py повторяют классы внутри
# datasets.py и generator.py (но класс генератора оказался необходим, потому что без него
# возникала ошибка "Can't get attribute..."). Основу для интерфейса я взял отсюда:
# https://wiki.programstore.ru/python-rabota-s-izobrazheniyami-v-tkinter/.

import tkinter
import tkinter.messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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

class GAN_Dataset(Dataset):
  def __init__(self, file):
    super().__init__()
    self.file = file
    self.len_ = 1

  def __len__(self):
      return self.len_

  def load_sample(self, file):
    image = Image.open(file)
    image.load()
    width, height = image.size
    facade = image.crop((0, 0, width//2, height))
    segment = image.crop((width//2, 0, width, height))
    cropper = transforms.RandomCrop(size=(256, 256))
    facade = cropper(facade.resize((286, 286)))
    segment = cropper(segment.resize((286, 286)))
    return facade, segment

  def prepare_data(self, image):
    image = np.array(image)
    image = np.array(image / 255, dtype='float32')
    return image

  def __getitem__(self, index):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    target, input = self.load_sample(self.file)
    input = self.prepare_data(input)
    target = self.prepare_data(target)
    input = transform(input)
    target = transform(target)
    return target, input

class App:
    def __init__(self):
        self.root = tkinter.Tk()
        self.root.title('Генерация фасадов')
        self.frame = tkinter.Frame(self.root)
        self.frame.grid()

        self.label = tkinter.Label(self.frame, text="Укажите путь до исходного изображения:").grid(row=1, column=1)
        self.entry = tkinter.Entry(self.frame)
        self.entry.grid(row=1, column=2, sticky=tkinter.NSEW)

        self.addition = tkinter.Button(self.frame, text="Добавить", command=self.add).grid(row=1, column=3)
        self.generation = tkinter.Button(self.frame, text="Сгенерировать", command=self.generate).grid(row=1, column=4)

        self.canvas_1 = tkinter.Canvas(self.frame, height=256, width=256)
        self.canvas_1.grid(row=2, column=1)
        self.canvas_2 = tkinter.Canvas(self.frame, height=256, width=256)
        self.canvas_2.grid(row=2, column=2)
        self.root.mainloop()

    def add(self):
        directory = self.entry.get()
        try:
            self.in_img = Image.open(directory)
            width, height = self.in_img.size
            self.data = DataLoader(GAN_Dataset(directory), shuffle=True, batch_size=1)
            self.in_img = self.in_img.crop((width//2, 0, width, height))
            self.in_photo = ImageTk.PhotoImage(self.in_img)
            self.c_1_image = self.canvas_1.create_image(0, 0, anchor='nw', image=self.in_photo)
        except:
            tkinter.messagebox.showerror("Генерация изображений", "Ошибка!")

    def generate(self):
        try:
            _, segment = next(iter(self.data))
            if torch.cuda.is_available():
                device = torch.device('cuda')
                segment.cuda()
            else:
                device = torch.device('cpu')
            gen = torch.load('./model3_g.pth', map_location=device)
            self.out_img = gen(segment).detach().cpu()
            self.out_img = self.out_img[0]
            self.out_img = self.out_img.numpy().transpose((1, 2, 0))
            mean = np.array([0.5, 0.5, 0.5])
            std = np.array([0.5, 0.5, 0.5])
            self.out_img = std * self.out_img + mean
            self.out_img = np.clip(self.out_img, 0, 1)
            self.out_img = Image.fromarray((255*self.out_img).astype(np.uint8))
            self.out_photo = ImageTk.PhotoImage(self.out_img)
            self.c_2_image = self.canvas_2.create_image(0, 0, anchor='nw', image=self.out_photo)
        except:
            tkinter.messagebox.showerror("Генерация изображений", "Ошибка!")

app = App()