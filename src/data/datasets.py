# Код файла для загрузки и предобработки датасета datasets.py взят из домашнего задания по
# классификации Симпсонов. Я не разобрался, как исполнять bash-скрипты на локальном компьютере,
# поэтому мне пришлось скачать архив с датасетом facades.zip. Небольшая ремарка: обычно из датасета
# получают батчи в форме x_batch, y_batch. Я же x_batch и y_batch поменял местами (то есть получение
# батча выглядит так: from facade, segment in d_loader...).
# Объём тренировочного датасета - 400 картинок, валидационного и тестового - 100 и 106 картинок
# соответственно.

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import PIL
import numpy as np
import zipfile
from pathlib import Path

z = zipfile.ZipFile('facades.zip')
z.extractall('facades')
z.close()

TRAIN_DIR = Path('facades/train')
VAL_DIR = Path('facades/val')
TEST_DIR = Path('facades/test')

train_files = list(TRAIN_DIR.rglob('*.jpg'))
val_files = list(VAL_DIR.rglob('*.jpg'))
test_files = list(TEST_DIR.rglob('*.jpg'))

class GAN_Dataset(Dataset):
  def __init__(self, files):
    super().__init__()
    self.files = sorted(files)
    self.len_ = len(self.files)

  def __len__(self):
    return self.len_

  def load_sample(self, file):
    image = PIL.Image.open(file)
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
    target, input = self.load_sample(self.files[index])
    input = self.prepare_data(input)
    target = self.prepare_data(target)
    input = transform(input)
    target = transform(target)
    return target, input

def imshow(inp, plt_ax=plt, default=False):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt_ax.imshow(inp)
    plt_ax.grid(False)

def view_sample(img_1, img_2):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    imshow(img_1, plt_ax=ax.flatten()[0])
    imshow(img_2, plt_ax=ax.flatten()[1])
    plt.show()

batch_size = 1

val_ds = GAN_Dataset(val_files)
train_ds = GAN_Dataset(train_files)
test_ds = GAN_Dataset(test_files)
train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(val_ds, shuffle=False, batch_size=batch_size)
test_loader = DataLoader(test_ds, shuffle=False, batch_size=batch_size)

# random_sample = np.random.randint(100)
# facade, segment = train_ds[random_sample]
# view_sample(facade, segment)
