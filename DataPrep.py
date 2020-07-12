import numpy as np
import cv2
import imgaug as ia
import imgaug.augmenters as iaa

augment_pipeline = iaa.Sequential([
    iaa.Affine(rotate=(-15, 15)),
    iaa.Affine(shear=(-15, 15)),
    iaa.Crop(percent=(0, 0.25)),
    iaa.Grayscale(alpha=(0, 1))
])
def fit_image(img, expected_shape=(478, 968)):
    HEIGHT, WIDTH = expected_shape
    height, width = img.shape[:2]

    if WIDTH / width < HEIGHT / height:
        ratio = WIDTH / width
    else:
        ratio = HEIGHT / height

    blank_img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    img = cv2.resize(img, None, fx=ratio, fy=ratio)
    height, width = img.shape[:2]
#     blank_img[HEIGHT // 2 - height // 2: HEIGHT // 2 + height // 2,
#              WIDTH // 2 - width // 2: WIDTH // 2 + width // 2] = img
    blank_img[(HEIGHT - height) // 2: (HEIGHT + height) // 2,
             (WIDTH - width) // 2: (WIDTH + width) // 2] = img
    img = blank_img
    return img

def preprocess_image(img):
    img = fit_image(img)
    img = img[:, :, ::-1]
    img = img / 255.
    return img