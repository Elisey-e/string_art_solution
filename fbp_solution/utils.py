import numpy as np
import matplotlib.pyplot as plt


def show_image(img, title='', **kwargs):
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray', vmin=0, vmax=1, **kwargs)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def show_side_by_side(img_a, img_b, title_a='', title_b='', figsize=(10, 4.5)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(img_a, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(title_a)
    axes[0].axis('off')
    axes[1].imshow(img_b, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(title_b)
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()


def show_sinogram(data, angles, title='', cmap='gray', colorbar_label='', **kwargs):
    plt.figure(figsize=(8, 4))
    plt.imshow(data, cmap=cmap, aspect='auto',
               extent=[0, data.shape[1], angles[-1], 0], **kwargs)
    plt.xlabel('Смещение ρ (пиксели)')
    plt.ylabel('Угол φ (градусы)')
    plt.title(title)
    if colorbar_label:
        plt.colorbar(label=colorbar_label)
    plt.tight_layout()
    plt.show()


def show_binary_mask(data, angles, title=''):
    plt.figure(figsize=(8, 4))
    plt.imshow(data, cmap='gray', aspect='auto',
               extent=[0, data.shape[1], angles[-1], 0])
    plt.xlabel('Смещение ρ (пиксели)')
    plt.ylabel('Угол φ (градусы)')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def show_threads_on_black(preview, title=''):
    plt.figure(figsize=(6, 6))
    plt.imshow(preview, cmap='gray', vmin=0, vmax=1)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
