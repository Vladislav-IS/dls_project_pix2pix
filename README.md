# dls_project_pix2pix
Итоговый проект, тема - "Генерация изображений".

На всякий случай приведу данные о себе: ФИО - Семенов Владислав Юрьевич, Stepik-ID - 273574954.

В данном проекте решалась задача генерации фасадов зданий по их сегментированным изображениям. Использовалась архитектура pix2pix. К сожалению, из-за нехватки времени (а также из-за того, что в первую половину проекта я пытался генерировать не фасады, а обувь по контурам) мне не удалось реализовать полноценную нейросеть: на сгенерированных изображениях видны неточности и многочисленные артефакты. Да еще и собственную задачу я выбрать не успел. Тем не менее обучение модели и интерфейс для генерации более-менее работают.

![image](https://user-images.githubusercontent.com/74904348/123541159-ac68cb00-d74b-11eb-9b85-81e3bb3c59e2.png)

![image](https://user-images.githubusercontent.com/74904348/123542195-ed171300-d750-11eb-8e96-5d4ca380ae2c.png)

![image](https://user-images.githubusercontent.com/74904348/123542231-20f23880-d751-11eb-8149-2cbdd00d1ab2.png)

Теперь расскажу о каждом файле отдельно (информацию ниже приведу также в комментариях внутри файлов):
1. generator.py и discriminator.py содержат код генератора и дискриминатора соответственно. При написании нейросети я опирался на оригинальную статью (https://arxiv.org/pdf/1611.07004.pdf) и на это видео с Ютуба: https://youtu.be/SuddDSqGRzg. Из видео я взял идею создать отдельные классы слоёв нейросетей, на основе которых создаются классы самих сетей. Генератор представляет собой укороченный UNet, а дискриминатор подобен патч-дискриминатору из оригинальной статьи (разве что на его выходе получается тензор 2х2, а не 70х70) и принимает на вход конкатенированный тензор "сегментированное изображение/оригинальный фасад" или "сегментированное изображение/сгенерированный фасад".
2. fit.py содержит код обучения GAN. Код я брал из семинарского ноутбука по генеративно-состязательным сетям (не знаю, может ли это считаться плагиатом). В качестве лосса использовалась кросс-энтропия, как в том же видео (https://youtu.be/SuddDSqGRzg). Да это и логично, учитывая, что BCELoss больше всего похож на оригинальный cGAN-лосс из статьи. В конце обучения функция fit() записывает два файла, содержащие дискриминатор и генератор (в нашем случае это файлы model3_g.pth, model3_d.pth). Не знаю, портит ли работу данного файла использование tqdm, но код я оставил таким, каким он был в Колабе. 
3. Код файла для загрузки и предобработки датасета datasets.py взят из домашнего задания по классификации Симпсонов. Я не разобрался, как исполнять bash-скрипты на локальном компьютере, поэтому мне пришлось скачать  архив с датасетом facades.zip. Небольшая ремарка: обычно из датасета получают батчи в форме x_batch, y_batch. Я же x_batch и y_batch поменял местами (то есть получение батча выглядит так: from facade, segment in d_loader...). datasets.py даёт возможность взглянуть на изображения в датасете, но строчки с кодом, которые позволяют это сделать, я закомментировал.
4. generate.py представляет собой полностью автономный файл с интерфейсом для генерации изображений (хотя ему все равно необходим файл model3_g.pth с моделью генератора). Из тех же соображений автономности классы внутри generate.py повторяют классы внутри datasets.py и generator.py (но класс генератора оказался необходим, потому что без него возникала ошибка "Can't get attribute..."). Основу для интерфейса я взял отсюда: https://wiki.programstore.ru/python-rabota-s-izobrazheniyami-v-tkinter/. Работает генерация следующим образом: необходимо, чтобы в одной папке находилась модель, сохранённая в pth-файле, и файл генерации. Изображения могут находиться в любом каталоге, но есть два ограничения: нужно указывать полный путь до изображения, если оно не находится в одной папке с файлом генерации; приходится загружать картинки как в оригинальном датасете (то есть те, у которых одна половина - фасад, другая половина - сегментированное изображение). После того, как в текстовое поле вводится путь до изображения, нажимается кнопка "Добавить". После этого в окне интерфейса появляется сегментированное изображение. Далее нажимается кнопка "Сгенерировать", и справа от исходной картинки появляется сгенерированная.
5. В папке с generate.py я положил несколько картинок .jpg.
6. Скрипт download_pix2pix_dataset.sh я честно позаимствовал отсюда: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master/datasets. Но его я только в Колабе использовал.

Те файлы, что не смгли уместиться в Гитхаб (а это архив с датасетом и обученные модели), можно найти по ссылкам в документе other.txt.

Это все, что я хотел сказать. Видимо, я немного переоценил свои силы (я всё-таки с базового потока), но что есть - то есть.

Бонусом приведу гифку, на которой запечатлён процесс генерации.

![Генерация-фасадов-2021-06-27-14-48-22](https://user-images.githubusercontent.com/74904348/123543406-e9868a80-d756-11eb-9247-62b5e4664e80.gif)
