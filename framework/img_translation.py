from PIL import Image
import numpy as np
from method_logger import *

ML = method_log()
img_path: str = '../images/cat.png'

def translate_image_to_rgb_matrix(img_path: str) -> np.ndarray:
    ML.start(func_name='translate_image_to_rgb_matrix', args={'img_path': type(img_path)})

    image = Image.open(img_path)
    image_rgb = image.convert('RGB')
    width, height = image_rgb.size

    log(f"Image Size: ({width}, {height}).")

    rgb_matrix: np.ndarray = np.zeros((width, height, 3), dtype=np.uint8)

    for i in range(width):
        for j in range(height):
            r, g, b = image_rgb.getpixel((j, i))
            rgb_matrix[i, j] = [r, g, b]

    ML.end(status=1, return_val=rgb_matrix)
    return rgb_matrix

rgb_matrix = translate_image_to_rgb_matrix(img_path)
log(f"RGB Matrix: {rgb_matrix[1][1]}")