# fit.py содержит код обучения GAN. Код я брал из семинарского ноутбука по
# генеративно-состязательным сетям (не знаю, может ли это считаться плагиатом).
# В качестве лосса использовалась кросс-энтропия, как в том же видео (https://youtu.be/SuddDSqGRzg).
# Да это и логично, учитывая, что BCELoss больше всего похож на оригинальный cGAN-лосс из статьи.
# В конце обучения функция fit() записывает два файла, содержащие дискриминатор и генератор
# (в нашем случае это файлы model3_g.pth, model3_d.pth). Не знаю, портит ли работу данного файла
# использование tqdm, но код я оставил таким, каким он был в Колабе.
# Обучение заняло 350 эпох.

import discriminator
import generator
import datasets
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = {'discriminator': discriminator.Discriminator().to(device),
		 'generator': generator.Generator().to(device)}

epochs = 350

def fit(model, epochs):
	model['discriminator'].train()
	model['generator'].train()
	torch.cuda.empty_cache()

	bce = nn.BCELoss()
	l1 = nn.L1Loss()
	losses_g = []
	losses_d = []
	real_scores = []
	fake_scores = []

	lr = 0.0002
	optimizer = {"discriminator": torch.optim.Adam(model["discriminator"].parameters(),
												   lr=lr, betas=(0.5, 0.999)),
				 "generator": torch.optim.Adam(model["generator"].parameters(),
											   lr=lr, betas=(0.5, 0.999))}

	for epoch in range(epochs):
		loss_d_per_epoch = []
		loss_g_per_epoch = []
		real_score_per_epoch = []
		fake_score_per_epoch = []
		for facades, segments in tqdm(datasets.train_loader):
			if torch.cuda.is_available():
				facades = facades.cuda()
				segments = segments.cuda()
			optimizer["discriminator"].zero_grad()
			real_preds = model["discriminator"](segments, facades)
			real_targets = torch.ones_like(real_preds, device=device) - 0.05 * torch.randn_like(real_preds,
																								device=device)
			real_loss = bce(real_preds, real_targets)
			cur_real_score = torch.mean(real_preds).item()

			fake_images = model["generator"](segments)
			fake_preds = model["discriminator"](segments, fake_images)
			fake_targets = torch.zeros_like(fake_preds, device=device) + 0.05 * torch.randn_like(fake_preds,
																								 device=device)
			fake_loss = bce(fake_preds, fake_targets)
			cur_fake_score = torch.mean(fake_preds).item()

			real_score_per_epoch.append(cur_real_score)
			fake_score_per_epoch.append(cur_fake_score)

			loss_d = real_loss + fake_loss
			loss_d.backward()
			optimizer["discriminator"].step()
			loss_d_per_epoch.append(loss_d.item())

			optimizer["generator"].zero_grad()

			fake_images = model["generator"](segments)

			preds = model["discriminator"](segments, fake_images)
			targets = torch.ones_like(preds, device=device) - 0.05 * torch.randn_like(preds, device=device)
			loss_g = bce(preds, targets) + 100 * l1(preds, targets)

			loss_g.backward()
			optimizer["generator"].step()
			loss_g_per_epoch.append(loss_g.item())

			losses_g.append(np.mean(loss_g_per_epoch))
			losses_d.append(np.mean(loss_d_per_epoch))
			real_scores.append(np.mean(real_score_per_epoch))
			fake_scores.append(np.mean(fake_score_per_epoch))

		print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
			epoch + 1, epochs,
			losses_g[-1], losses_d[-1], real_scores[-1], fake_scores[-1]))

	torch.save(model['generator'], 'model400_g.pth')
	torch.save(model['discriminator'], 'model400_d.pth')

fit(model, epochs)