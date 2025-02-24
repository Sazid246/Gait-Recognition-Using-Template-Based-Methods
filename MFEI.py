import os
import numpy as np
import cv2
from skimage.transform import resize
from matplotlib import pyplot as plt

def image_extract(img, newsize):
    """
    Extracts the non-zero portion of the image and resizes it to a fixed size.
    Args:
        img (ndarray): Input image.
        newsize (tuple): Desired output size (height, width).
    Returns:
        ndarray: Resized non-zero region of the image.
    """
    if img.mean() == 0:  # Check if the image is blank
        return np.zeros(newsize)

    non_zero_columns = np.where(img.mean(axis=0) != 0)[0]
    non_zero_rows = np.where(img.mean(axis=1) != 0)[0]

    if len(non_zero_columns) == 0 or len(non_zero_rows) == 0:
        return np.zeros(newsize)

    x_s = non_zero_columns.min()
    x_e = non_zero_columns.max()
    y_s = non_zero_rows.min()
    y_e = non_zero_rows.max()

    # Crop and resize
    cropped_img = img[y_s:y_e, max(0, x_s):min(x_e, img.shape[1])]
    img_resized = resize(cropped_img, newsize, anti_aliasing=True)
    return img_resized


def create_mfei(images):
    """
    Computes the Motion Flow Energy Image (MFEI) from a list of frames.
    Args:
        images (list): List of grayscale images (2D numpy arrays).
    Returns:
        ndarray: Motion Flow Energy Image.
    """
    num_frames = len(images)
    if num_frames < 2:
        raise ValueError("At least two frames are required to compute MFEI.")

    # Initialize MFEI as a float array with the same shape as the input images
    mfei = np.zeros_like(images[0], dtype=np.float32)

    for i in range(num_frames - 1):
        # Compute absolute difference between consecutive frames (|I_(i+1) - I_i|)
        motion_image = np.abs(images[i + 1].astype(np.float32) - images[i].astype(np.float32))
        mfei += motion_image

    # Average the motion images
    mfei /= (num_frames - 1)
    return mfei


def process_folder(input_folder, output_folder_base):
    """
    Processes all subfolders, extracts frames, computes MFEI, and saves the output images.
    Args:
        input_folder (str): Path to the folder containing input frames.
        output_folder_base (str): Path to the folder where MFEI outputs will be saved.
    """
    for subdir, _, files in os.walk(input_folder):
        images = []
        for f in sorted(files):  # Sort files to maintain frame order
            if f.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(subdir, f)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Extract non-zero region and resize
                    img_processed = image_extract(img, (128, 64))
                    images.append(img_processed)

        if len(images) < 2:  # Skip if there are less than 2 images
            continue

        # Compute MFEI
        mfei = create_mfei(images)

        # Prepare output folder
        relative_path = os.path.relpath(subdir, input_folder)
        output_folder = os.path.join(output_folder_base, relative_path)
        os.makedirs(output_folder, exist_ok=True)

        # Save the MFEI image
        output_file = os.path.join(output_folder, 'mfei_output.png')
        plt.imshow(mfei, cmap='gray')
        plt.axis('off')
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        plt.close()


# Define input and output folders
input_folder = '/home/sazid/Model_Based_GAIT_PROJECT/casia_sl/'  # Input folder path
output_folder_base = '/home/sazid/Model_Based_GAIT_PROJECT/Pre_Processing/output/MFEI2_Image/'  # Output folder path

# Process the input folder
process_folder(input_folder, output_folder_base)
