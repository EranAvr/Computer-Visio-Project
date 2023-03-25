import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from PIL import Image


########################################################################
#                  BASIC                                               #
########################################################################

def aug_to_PIL():
    # to PIL:
    PIL_img = T.ToPILImage()(img_tensor)
    print("PIL_img class:", PIL_img.__class__, "\tPIL_img shape:",
          np.array(PIL_img).shape)
    plt.imshow(PIL_img)
    plt.title('ToPILImage')
    plt.show()

def aug_resize():
    # Resize:
    resized_imgs = [T.Resize(size)(img_tensor) for size in (128, 256)]
    resized_imgs_titles = ['Resize(128)', 'Resize(256)']
    print("resized_img class:", resized_imgs[0].__class__, "\tresized_img shape:",
          resized_imgs[0].shape)
    fig, axs = plt.subplots(nrows=1, ncols=2)
    for i in range(len(resized_imgs)):
        axs[i].imshow(resized_imgs[i].permute(1, 2, 0))
        axs[i].set_title(resized_imgs_titles[i])
    plt.show()

def aug_center_crop():
    # Center Crop:
    CROPPING_SIZES =(75, 100, 400)
    center_crop_imgs = [T.CenterCrop(size=size)(img_tensor) for size in CROPPING_SIZES]
    center_crop_titles = ['CenterCrop({})'.format(size) for size in CROPPING_SIZES]
    fig, axs = plt.subplots(nrows=1, ncols=3)
    for i, ax in enumerate(axs):
        ax.imshow(center_crop_imgs[i].permute(1, 2, 0))
        ax.set_title(center_crop_titles[i])
    plt.show()

def aug_random_crop():
    # Random Crop:
    CROPPING_SIZES =(150, 200, 400)
    random_crop_imgs = [T.RandomCrop(size=size)(img_tensor) for size in CROPPING_SIZES]
    random_crop_titles = ['RandomCrop({})'.format(size) for size in CROPPING_SIZES]
    fig, axs = plt.subplots(nrows=1, ncols=3)
    for i, ax in enumerate(axs):
        ax.imshow(random_crop_imgs[i].permute(1, 2, 0))
        ax.set_title(random_crop_titles[i])
    plt.show()

def aug_normalize():
    # Normalization:
    normalized_img = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])(
        img_tensor.to(torch.float))
    print("normalized_img class:", normalized_img.__class__,
          "\tnormalized_img shape:", normalized_img.shape)
    plt.imshow(normalized_img.permute(1, 2, 0))
    plt.title('Normalize()')
    plt.show()

def aug_rotate():
    # Rotate:
    rotated_imgs = [T.RandomRotation(degrees=degree)(img_tensor) for degree in (90, 180, 270, 300)]
    rotated_imgs_titles = ['Rotated({})'.format(degree) for degree in (90, 180, 270, 300)]

    fig, axs = plt.subplots(nrows=2, ncols=2)
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(rotated_imgs[i].permute(1, 2, 0))
        ax.set_title(rotated_imgs_titles[i])
    plt.show()

def aug_gaussian_blur():
    # Gaussian Blur:
    KERNEL_SIZE = (31,31)
    SIGMAS = (2, 5, 7, 10)
    gaussian_blur_imgs = [T.GaussianBlur(kernel_size=KERNEL_SIZE, sigma=sigma)(img_tensor) for sigma in SIGMAS]
    gaussian_blur_imgs_titles = ['Gaussian Blur sigma={}'.format(sig) for sig in SIGMAS]
    fig, axs = plt.subplots(nrows=2, ncols=2)
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(gaussian_blur_imgs[i].permute(1, 2, 0))
        ax.set_title(gaussian_blur_imgs_titles[i])
    plt.show()

class RandomGaussianBlur(nn.Module):

    def __init__(self, kernel, sigma, probability):
        super().__init__()
        self.kernel = kernel
        self.sigma = sigma
        self.p = probability

    def forward(self, img):
        if random.random() > self.p: # catch 'bad' probability and return original image
            return img
        return T.GaussianBlur(kernel_size=self.kernel, sigma=self.sigma)(img)


########################################################################
#                  ADVANCED                                            #
########################################################################

# Noise:
class UniformNoiseTransform(nn.Module):

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, img):
        uniform_noise = torch.rand_like(img, dtype=torch.float)
        noisy_output = torch.clip((img / 255) + uniform_noise * self.factor, 0., 1.)
        return noisy_output

class GaussianNoiseTransform(nn.Module):

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, img):
        gaussian_noise = torch.randn_like(img, dtype=torch.float)
        noisy_output = torch.clip((img / 255) + gaussian_noise * self.factor,
                                  0., 1.)
        return noisy_output

class RandomUniformNoiseTransform(nn.Module):

    def __init__(self, factor, probability):
        super().__init__()
        self.factor = factor
        self.p = probability

    def forward(self, img):
        if random.random() > self.p: # catch 'bad' probability and return original image
            return img

        uniform_noise = torch.rand_like(img, dtype=torch.float)
        if img.max().item() > 1.0:  # in case pixels values exceeds '1'
            img = img / 255
        noisy_output = torch.clip((img) + uniform_noise * self.factor, 0., 1.)
        return noisy_output

class RandomGaussianNoiseTransform(nn.Module):

    def __init__(self, factor, probability):
        super().__init__()
        self.factor = factor
        self.p = probability

    def forward(self, img):
        if random.random() > self.p: # catch 'bad' probability and return original image
            return img

        gaussian_noise = torch.randn_like(img, dtype=torch.float) # randn func uses normal-dist
        if img.max().item() > 1.0:  # in case pixels values exceeds '1'
            img = img / 255
        noisy_output = torch.clip((img) + gaussian_noise * self.factor,
                                  0., 1.)
        return noisy_output


# Black-Boxes:
def add_black_boxes(input_img: torch.Tensor, box_size: int = 10, box_count: int = 1):
    rows_idx_lim = input_img.size()[-2] - box_size
    cols_idx_lim = input_img.size()[-1] - box_size
    output_img = torch.clone(input_img).detach()
    for b in range(box_count):
        i = torch.randint(rows_idx_lim, [1])
        j = torch.randint(cols_idx_lim, [1])
        for channel in range(input_img.size()[0]):
            output_img[channel, i:i+box_size, j:j+box_size] = 0
    return output_img

def aug_black_boxes():
    img_bb = add_black_boxes(img_tensor, box_size=50, box_count=5)
    plt.imshow(img_bb.permute(1, 2, 0))
    plt.show()

# Central Region:
def add_black_ragion(input_img: torch.Tensor, region_radius: int = 50):
    height, width = input_img.size()[1:3]
    h_mid, w_mid = height // 2, width // 2
    output_img = torch.clone(input_img).detach()
    for channel in range(input_img.size()[0]):
        output_img[channel, h_mid - region_radius:h_mid + region_radius,
        w_mid - region_radius:w_mid + region_radius] = 0
    return output_img

def aug_central_region():
    img_cr = add_black_ragion(img_tensor, region_radius=100)
    plt.imshow(img_cr.permute(1, 2, 0))
    plt.show()


########################################################################
#                  MODULE Test Script                                  #
########################################################################


if __name__ == '__main__':
    # Original:
    img = Image.open("D:\\Datasets\\good_citrus_dataset_cut\\train\\black-spot\\black-spot (1).jpg")
    img = np.array(img)
    img_tensor = torch.tensor(img).permute(2, 0, 1)
    print("img class:", img.__class__, "\timg shape:", img.shape)
    print("img_tensor class:", img_tensor.__class__, "\timg_tensor shape:", img_tensor.shape)

    MAGNITUDES = (0.2, 0.5, 0.7)
    noisy_imgs = [RandomUniformNoiseTransform(factor, 1.0)(img_tensor) for factor in MAGNITUDES] \
                 + [RandomGaussianNoiseTransform(factor, 1.0)(img_tensor) for factor in
                    MAGNITUDES]
    noisy_titles = ['Uniform Noise factor={}'.format(factor) for factor in
                    MAGNITUDES] \
                   + ['Gaussian Noise factor={}'.format(factor) for factor in
                      MAGNITUDES]
    fig, axs = plt.subplots(nrows=2, ncols=3)
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(noisy_imgs[i].permute(1, 2, 0))
        ax.set_title(noisy_titles[i])
    plt.show()



