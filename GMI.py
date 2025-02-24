import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from imageio import imread
from skimage.transform import resize

def image_extract(img, newsize):
    if img.mean() == 0:
        return np.zeros(newsize)

    non_zero_columns = np.where(img.mean(axis=0) != 0)[0]
    non_zero_rows = np.where(img.mean(axis=1) != 0)[0]

    if len(non_zero_columns) == 0 or len(non_zero_rows) == 0:
        return np.zeros(newsize)

    x_s = non_zero_columns.min()
    x_e = non_zero_columns.max()

    y_s = non_zero_rows.min()
    y_e = non_zero_rows.max()

    img = img[y_s:y_e, max(0, x_s):min(x_e, img.shape[1])]
    img_resized = resize(img, newsize, anti_aliasing=True)
    return img_resized

def create_gmi(images):
    gmi = np.zeros_like(images[0], dtype=np.float32)
    for img in images:
        gmi += img.astype(np.float32)
    gmi /= len(images)
    return gmi

def process_folder(input_folder, output_folder_base):
    image_counters = {}
    for subdir, _, files in os.walk(input_folder):
        if not files:
            continue
        images = [imread(os.path.join(subdir, f)) for f in files if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        if not images:
            continue

        images = [image_extract(img, (128, 64)) for img in images]

        if not images or all(img.size == 0 for img in images):
            continue

        gmi = create_gmi(images)

        relative_path = os.path.relpath(subdir, input_folder)
        output_folder = os.path.join(output_folder_base, relative_path)
        os.makedirs(output_folder, exist_ok=True)

        if relative_path not in image_counters:
            image_counters[relative_path] = 1
        output_file = os.path.join(output_folder, f'gmi_output_{image_counters[relative_path]}.png')

        plt.imshow(gmi, cmap='gray')
        plt.axis('off')
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        plt.close()

        image_counters[relative_path] += 1

input_folder = '/home/sazid/Model_Based_GAIT_PROJECT/casia_sl/'
output_folder_base = '/home/sazid/Model_Based_GAIT_PROJECT/Pre_Processing/output/GMI_Image/'
process_folder(input_folder, output_folder_base)
