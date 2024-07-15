import numpy as np
from log import *

ML = method_log()

def multiply_matrices(matrix_1: np.array, matrix_2: np.array) -> np.ndarray:
    ML.start(func_name='multiply_matrices', args={'matrix_1': type(matrix_1.shape), 'matrix_2': type(matrix_2.shape)})

    matrix_1_col_size = matrix_1.shape[1]
    matrix_2_row_size = matrix_2.shape[0]

    is_size_equal = (matrix_1_col_size == matrix_2_row_size)
    log(f"CONDITION: is_size_equal = (matrix_1_col_size == matrix_2_row_size) = {is_size_equal}")

    if is_size_equal:
        product = np.matmul(matrix_1, matrix_2)
        ML.end(status=1, return_val=product)
        return product
    elif not is_size_equal:
        ML.end(status=0, return_val=None)
        return None

def generate_kernel(values_range: int) -> np.ndarray:
    values_range = [-values_range, values_range]


np.random.seed(1)

matrix_1 = np.random.randint(50, 201, size=(6, 7))
matrix_2 = np.random.randint(50, 201, size=(7, 9))
multiply_matrices(matrix_1, matrix_2)