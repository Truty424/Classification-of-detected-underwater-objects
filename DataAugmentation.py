import imageio.v3 as imageio
import numpy as np
import imgaug.augmenters as iaa
from PIL import Image

img_num = 4

input_img = imageio.imread('/Users/adam/PycharmProjects/DataAugmentation/InputImages/WZ08:39_1.png')

images = np.array(
    [input_img for _ in range(img_num)],
    dtype=np.uint8
)

seq = iaa.Sequential([
    iaa.Affine(
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-180, 180),
        shear=(-20, 20),
        scale=(0.75, 1.3)
    ),

    iaa.Fliplr(0.5),

    iaa.Flipud(0.4),

    iaa.Multiply((0.8, 1.2), per_channel=0.2),

], random_order=True)


seq2 = iaa.Sequential([
    iaa.Sometimes(
        0.5,
        iaa.AdditiveGaussianNoise(scale=(0, 0.3 * 255)),
        iaa.LinearContrast((0.75, 1.5)),
    )

], random_order=True)

images_aug = seq(images=images)
images_aug2 = seq2(images=images_aug)
for i, img in enumerate(images_aug2):
    im = Image.fromarray(img)
    im.save(f'/Users/adam/PycharmProjects/DataAugmentation/Images/{i}.png', format='PNG')



