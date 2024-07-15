import numpy as np
from method_logger import ML, log
from img_translation import *

class KernelConfig:
    def __init__(self):
        self.weight: float = .1
        self.size: int = 3
        self.num: int = 16
        self.stride: int = 1

KC = KernelConfig()

def multiply_matrices(matrix_1: np.array, matrix_2: np.array) -> np.ndarray:
    ML.start(func_name='multiply_matrices', args={'matrix_1': type(matrix_1.shape), 'matrix_2': type(matrix_2.shape)})

    matrix_1_col_size = matrix_1.shape[1]
    matrix_2_row_size = matrix_2.shape[0]

    is_size_equal = (matrix_1_col_size == matrix_2_row_size)
    log(f"VARIABLE   is_size_equal = (matrix_1_col_size == matrix_2_row_size) = {is_size_equal}")

    if is_size_equal:
        product = np.matmul(matrix_1, matrix_2)
        ML.end(status=1, return_val=product)
        return product
    elif not is_size_equal:
        ML.end(status=0, return_val=None)
        return None

# S. Sabyasachi. (2018) Deciding optimal kernel size for CNN. https://towardsdatascience.com/deciding-optimal-filter-size-for-cnns-d6f7b56f9363
def generate_kernels(kernel_config: dict = KC) -> np.ndarray:
    ML.start(func_name='generate_kernels', args={'kernel_weight': type(KC.weight), 'kernel_size': type(KC.size), 'kernel_num': type(KC.num)})

    # Generates KC.num amount of kernels 3x3 kernels each with 3 channels for RGB, or simply 3x3x3x16.
    kernels = np.random.randn(KC.size, KC.size, 3, KC.num) * KC.weight
    log(f"VARIABLE   kernels.shape = {kernels.shape} | info.kernel_weight = {KC.weight}")


    ML.end(status=1, return_val=kernels)
    return kernels

def convolve_matrices(rgb_matrix: np.ndarray, kernels: np.ndarray, kernel_config: dict = KC) -> np.ndarray:
    ML.start(func_name='convolve_matrices', args={'rgb_matrix': type(rgb_matrix), 'kernels': type(kernels), 'kernel_config': type(KC)})

    input_height, input_width, _ = rgb_matrix.shape
    output_height: int = (input_height - KC.size) // KC.stride + 1
    output_width: int = (input_width - KC.size) // KC.stride + 1
    convolution_map: np.ndarray = np.zeros((output_height, output_width, KC.num))

    log(f"VARIABLE   convolution_map.shape = {convolution_map.shape}")
    ML.end(1, [[]])
    return [[]]

np.random.seed(1)
rand_matrix_1 = np.random.randint(50, 201, size=(6, 7))
rand_matrix_2 = np.random.randint(50, 201, size=(7, 9))
multiply_matrices(matrix_1=rand_matrix_1, matrix_2=rand_matrix_2)

img_path: str = '../images/cat.png'
test_rgb_matrix: np.ndarray = generate_RGB_matrix(img_path)
default_kernels: np.ndarray = generate_kernels()
convolve_matrices(rgb_matrix=test_rgb_matrix, kernels=default_kernels)