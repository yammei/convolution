<h1>Convolutional Neural Network Framework Project</h1>
Author Note: Most of what's left is all the back propogation. After which, I'll be testing on various image sizes to make sure everything works. Final objective is to train it on enough images to classify portraits of cats.

<h1>Project Progress</h1>
<i>Most recent terminal output.</i><br><br>

```bash
■ 1 ■      ■ 16:37:49 ■


METHOD     generate_RGB_matrix(img_path: str)
VARIABLE   width, height = (32, 32)
RETURN     Status: Success | Type: numpy.ndarray | Size: 3.216KB


■ 2 ■      ■ 16:37:49 ■


METHOD     generate_RGB_matrix(img_path: str)
VARIABLE   width, height = (32, 32)
RETURN     Status: Success | Type: numpy.ndarray | Size: 3.216KB


■ 3 ■      ■ 16:37:49 ■


METHOD     generate_RGB_matrix(img_path: str)
VARIABLE   width, height = (32, 32)
RETURN     Status: Success | Type: numpy.ndarray | Size: 3.216KB


■ 4 ■      ■ 16:37:49 ■


METHOD     generate_RGB_matrix()
VARIABLE   pooled_map.shape = (3, 32, 32, 3)
RETURN     Status: Success | Type: numpy.ndarray | Size: 9.376KB


■ 5 ■      ■ 16:37:49 ■


METHOD     generate_kernels(kernel_weight: float ...
           -, kernel_size: int, kernel_num: int)
VARIABLE   kernels.shape = (3, 3, 3, 16)
RETURN     Status: Success | Type: numpy.ndarray | Size: 3.616KB


■ 6 ■      ■ 16:37:49 ■


METHOD     convolution(rgb_matrix: numpy.nd ...
           -array, kernels: numpy.ndarray, kernel: KernelConfig)
VARIABLE   batch_feature_map.shape = (3, 30, 30, 16)
RETURN     Status: Success | Type: numpy.ndarray | Size: 345.760KB


■ 7 ■      ■ 16:37:49 ■


METHOD     pool(batch_feature_map: n ...
           -umpy.ndarray, pool_size: int, pool_mode: str)
VARIABLE   pooled_map.shape = (3, 15, 15, 16)
RETURN     Status: Success | Type: numpy.ndarray | Size: 86.560KB


■ 8 ■      ■ 16:37:49 ■


METHOD     flat(pooled_map: numpy.nd ...
           -array)
VARIABLE   flattened_map.shape = (3, 3600)
RETURN     Status: Success | Type: numpy.ndarray | Size: 0.128KB

┌------- START -------┐

■ 9 ■      ■ 16:37:49 ■


METHOD     dense(flattened_map: numpy ...
           -.ndarray, neurons: int, activation: relu)
VARIABLE   flattend_map.shape = (3, 3600) | weights.shape = (32, 3600) | weighted_map.shape = (3, 32)

┌----┐

■ 10 ■      ■ 16:37:49 ■


METHOD     relu(x: numpy.ndarray)
RETURN     Status: Success | Type: numpy.ndarray | Size: 0.896KB

└----┘

RETURN     Status: Success | Type: numpy.ndarray | Size: 0.896KB

└-------- END --------┘

┌------- START -------┐

■ 11 ■      ■ 16:37:49 ■


METHOD     dense(flattened_map: numpy ...
           -.ndarray, neurons: int, activation: softmax)
VARIABLE   flattend_map.shape = (3, 32) | weights.shape = (16, 32) | weighted_map.shape = (3, 16)

┌----┐

■ 12 ■      ■ 16:37:49 ■


METHOD     softmax(x: numpy.ndarray)
RETURN     Status: Success | Type: numpy.ndarray | Size: 0.512KB

└----┘

RETURN     Status: Success | Type: numpy.ndarray | Size: 0.512KB

└-------- END --------┘


■ 13 ■      ■ 16:37:49 ■


METHOD     back_propagation(predicted_output: nu ...
           -mpy.ndarray, labels: numpy.ndarray, weights: numpy.ndarray, activation: softmax)
VARIABLE   predicted_output.shape = (3, 16) | gradient_output.shape = (3, 16)
VARIABLE   gradient_input.shape = (3, 32) | gradient_weights.shape = (16, 32) | gradient_biases.shape = (1, 16)
RETURN     Status: Success | Type: tuple | Size: 0.064KB


■ 14 ■      ■ 16:37:49 ■


METHOD     update_parameters(weights: numpy.ndarr ...
           -ay, biases: numpy.ndarray, gradient_weights: numpy.ndarray, gradient_biases: numpy.ndarray, learning_rate: 0.001)
VARIABLE   updated_weights.shape = (16, 32) | updated_biases.shape = (1, 16)
RETURN     Status: Success | Type: tuple | Size: 0.056KB


■ 15 ■      ■ 16:37:49 ■


METHOD     back_propagation(predicted_output: nu ...
           -mpy.ndarray, labels: numpy.ndarray, weights: numpy.ndarray, activation: relu)
VARIABLE   predicted_output.shape = (3, 32) | gradient_output.shape = (3, 32)
VARIABLE   gradient_input.shape = (3, 3600) | gradient_weights.shape = (32, 3600) | gradient_biases.shape = (1, 32)
RETURN     Status: Success | Type: tuple | Size: 0.064KB


■ 16 ■      ■ 16:37:49 ■


METHOD     update_parameters(weights: numpy.ndarr ...
           -ay, biases: numpy.ndarray, gradient_weights: numpy.ndarray, gradient_biases: numpy.ndarray, learning_rate: 0.001)
VARIABLE   updated_weights.shape = (32, 3600) | updated_biases.shape = (1, 32)
RETURN     Status: Success | Type: tuple | Size: 0.056KB
```

<i>Previous terminal output.</i><br><br>

```bash

■ 1 ■                                                                           


METHOD     generate_RGB_matrix(img_path: str)
VARIABLE   width, height = (32, 32)
RETURN     Status: Success | Type: numpy.ndarray | Size: 3.216KB


■ 2 ■                                                                           


METHOD     generate_kernels(kernel_weight: float, kernel_size: int, kernel_num: int)
VARIABLE   kernels.shape = (3, 3, 3, 16)
RETURN     Status: Success | Type: numpy.ndarray | Size: 3.616KB


■ 3 ■                                                                           


METHOD     convolution(rgb_matrix: numpy.ndarray, kernels: numpy.ndarray, kernel: KernelConfig)
VARIABLE   feature_map.shape = (30, 30, 16)
RETURN     Status: Success | Type: numpy.ndarray | Size: 115.344KB


■ 4 ■                                                                           


METHOD     pool(feature_map: numpy.ndarray, pool_size: int, pool_mode: str)
VARIABLE   pooled_map.shape = (15, 15, 16)
RETURN     Status: Success | Type: numpy.ndarray | Size: 28.944KB


■ 5 ■                                                                           


METHOD     flat(pooled_map: numpy.ndarray)
VARIABLE   flattened_map.shape = (3600,)
RETURN     Status: Success | Type: numpy.ndarray | Size: 28.912KB


■ 6 ■                                                                           


METHOD     dense(flattened_map: numpy.ndarray, neurons: int, activation: none)
VARIABLE   flattend_map.shape = (3600,) | weights.shape = (64, 3600) | weighted_map.shape = (64,)
RETURN     Status: Success | Type: numpy.ndarray | Size: 0.624KB


■ 7 ■                                                                           


METHOD     relu(x: numpy.ndarray)
RETURN     Status: Success | Type: numpy.ndarray | Size: 0.624KB


■ 8 ■                                                                           


METHOD     dense(flattened_map: numpy.ndarray, neurons: int, activation: none)
VARIABLE   flattend_map.shape = (64,) | weights.shape = (32, 64) | weighted_map.shape = (32,)
RETURN     Status: Success | Type: numpy.ndarray | Size: 0.368KB


■ 9 ■                                                                           


METHOD     softmax(x: numpy.ndarray)
VARIABLE   sorted(probabilities[0:5]) = [0.0, 0.0, 0.0, 0.0, 0.0]
RETURN     Status: Success | Type: numpy.ndarray | Size: 0.368KB


■ Computation Details ■


+-----------------------+---------+---------------+-----------+
| Operation             |   Shape | DIM           |   CUM SUM |
|-----------------------+---------+---------------+-----------|
| generate_RGB_matrix() |       3 | (30, 30, 16)  |   2763.53 |
| generate_kernels()    |       4 | (3, 3, 3, 16) |      2.09 |
| convolution()         |       3 | (30, 30, 16)  |   2763.53 |
| pool()                |       3 | (15, 15, 16)  |   2223.13 |
| flat()                |       1 | (3600,)       |   2223.13 |
| dense()               |       1 | (64,)         |  71171.3  |
| dense()               |       1 | (32,)         |      1    |
+-----------------------+---------+---------------+-----------+

```

<h1>Dependencies</h1>

```bash
python3 -m venv myenv
source myenv/bin/activate

pip3 install numpy
pip3 install pillow
pip3 install tabulate
pip3 install matplotlib
```