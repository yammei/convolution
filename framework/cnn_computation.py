import numpy as np
from method_logger import *

ML = method_log()

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
def generate_kernels(kernel_weight: int = .1, kernel_size: int = 3, kernel_num: int = 16) -> np.ndarray:
    ML.start(func_name='generate_kernels', args={'kernel_weight': type(kernel_weight), 'kernel_size': type(kernel_size), 'kernel_num': type(kernel_num)})

    # Generates kernel_num amount of kernels 3x3 kernels each with 3 channels for RGB, or simply 3x3x3x16.
    kernels = np.random.randn(kernel_size, kernel_size, 3, kernel_num) * kernel_weight
    log(f"VARIABLE   kernels.shape = {kernels.shape}")

    ML.end(status=1, return_val=kernels)


np.random.seed(1)
matrix_1 = np.random.randint(50, 201, size=(6, 7))
matrix_2 = np.random.randint(50, 201, size=(7, 9))
multiply_matrices(matrix_1, matrix_2)

kernels = generate_kernels()