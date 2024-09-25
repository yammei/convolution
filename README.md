<h1>Convolutional Neural Network Framework Project</h1>
Author Note: Forward and backward pass finished implementing and successfully completes an epoch. Will be adding logs for backwards pass in next commit.

<h1>Project Progress</h1>
<i>Most recent terminal output. </i><br><br>

```bash
■ 1 ■      ■ 18:34:26 ■


METHOD     generate_RGB_matrix(img_path: str)
VARIABLE   width, height = (32, 32)
RETURN     Status: Success | Type: numpy.ndarray | Size: 3.216KB


■ 2 ■      ■ 18:34:26 ■


METHOD     generate_RGB_matrix(img_path: str)
VARIABLE   width, height = (32, 32)
RETURN     Status: Success | Type: numpy.ndarray | Size: 3.216KB


■ 3 ■      ■ 18:34:26 ■


METHOD     generate_RGB_matrix(img_path: str)
VARIABLE   width, height = (32, 32)
RETURN     Status: Success | Type: numpy.ndarray | Size: 3.216KB


■ 4 ■      ■ 18:34:26 ■


METHOD     generate_RGB_matrix()
VARIABLE   pooled_map.shape = (3, 32, 32, 3)
RETURN     Status: Success | Type: numpy.ndarray | Size: 9.376KB


■ 5 ■      ■ 18:34:26 ■


METHOD     generate_kernels(kernel_weight: float ...
           -, kernel_size: int, kernel_num: int)
VARIABLE   kernels.shape = (3, 3, 3, 16)
RETURN     Status: Success | Type: numpy.ndarray | Size: 3.616KB


■ 6 ■      ■ 18:34:26 ■


METHOD     convolution(rgb_matrix: numpy.nd ...
           -array, kernels: numpy.ndarray, kernel: KernelConfig)
VARIABLE   batch_feature_map.shape = (3, 30, 30, 16)
RETURN     Status: Success | Type: numpy.ndarray | Size: 345.760KB


■ 7 ■      ■ 18:34:26 ■


METHOD     pool(batch_feature_map: n ...
           -umpy.ndarray, pool_size: int, pool_mode: str)
VARIABLE   pooled_map.shape = (3, 15, 15, 16)
RETURN     Status: Success | Type: numpy.ndarray | Size: 86.560KB


■ 8 ■      ■ 18:34:26 ■


METHOD     flat(pooled_map: numpy.nd ...
           -array)
VARIABLE   flattened_map.shape = (3, 3600)
RETURN     Status: Success | Type: numpy.ndarray | Size: 0.128KB

┌------- START -------┐

■ 9 ■      ■ 18:34:26 ■


METHOD     dense(flattened_map: numpy ...
           -.ndarray, neurons: int, activation: relu)
VARIABLE   flattend_map.shape = (3, 3600) | weights.shape = (32, 3600) | weighted_map.shape = (3, 32)

┌----┐

■ 10 ■      ■ 18:34:26 ■


METHOD     relu(x: numpy.ndarray)
RETURN     Status: Success | Type: numpy.ndarray | Size: 0.896KB

└----┘

RETURN     Status: Success | Type: numpy.ndarray | Size: 0.896KB

└-------- END --------┘

┌------- START -------┐

■ 11 ■      ■ 18:34:26 ■


METHOD     dense(flattened_map: numpy ...
           -.ndarray, neurons: int, activation: softmax)
VARIABLE   flattend_map.shape = (3, 32) | weights.shape = (16, 32) | weighted_map.shape = (3, 16)

┌----┐

■ 12 ■      ■ 18:34:26 ■


METHOD     softmax(x: numpy.ndarray)
RETURN     Status: Success | Type: numpy.ndarray | Size: 0.512KB

└----┘

RETURN     Status: Success | Type: numpy.ndarray | Size: 0.512KB

└-------- END --------┘


■ 13 ■      ■ 18:34:26 ■


METHOD     back_propagation(predicted_output: nu ...
           -mpy.ndarray, labels: numpy.ndarray, weights: numpy.ndarray, activation: softmax)
VARIABLE   predicted_output.shape = (3, 16) | gradient_output.shape = (3, 16)
VARIABLE   gradient_input.shape = (3, 32) | gradient_weights.shape = (16, 32) | gradient_biases.shape = (1, 16)
RETURN     Status: Success | Type: tuple | Size: 0.064KB


■ 14 ■      ■ 18:34:26 ■


METHOD     update_weights_and_biases(weights: numpy.ndarr ...
           -ay, biases: numpy.ndarray, gradient_weights: numpy.ndarray, gradient_biases: numpy.ndarray, learning_rate: 0.001)
VARIABLE   updated_weights.shape = (16, 32) | updated_biases.shape = (1, 16)
RETURN     Status: Success | Type: tuple | Size: 0.056KB


■ 15 ■      ■ 18:34:26 ■


METHOD     back_propagation(predicted_output: nu ...
           -mpy.ndarray, labels: numpy.ndarray, weights: numpy.ndarray, activation: relu)
VARIABLE   predicted_output.shape = (3, 32) | gradient_output.shape = (3, 32)
VARIABLE   gradient_input.shape = (3, 3600) | gradient_weights.shape = (32, 3600) | gradient_biases.shape = (1, 32)
RETURN     Status: Success | Type: tuple | Size: 0.064KB


■ 16 ■      ■ 18:34:26 ■


METHOD     update_weights_and_biases(weights: numpy.ndarr ...
           -ay, biases: numpy.ndarray, gradient_weights: numpy.ndarray, gradient_biases: numpy.ndarray, learning_rate: 0.001)
VARIABLE   updated_weights.shape = (32, 3600) | updated_biases.shape = (1, 32)
RETURN     Status: Success | Type: tuple | Size: 0.056KB


■ 17 ■      ■ 18:34:26 ■


METHOD     back_propagation_conv(predicted_output: nu ...
           -mpy.ndarray, labels: numpy.ndarray, kernels: numpy.ndarray)
VARIABLE   derivative_predicted_output.shape = (3, 30, 30, 16) | derivative_kernels.shape = (3, 3, 3, 16)
RETURN     Status: Success | Type: tuple | Size: 0.056KB


■ 18 ■      ■ 18:34:26 ■


METHOD     updated_kernels(currnels: numpy.ndar ...
           -ray, gradient_kernels: numpy.ndarray, learning_rate: 0.001)
VARIABLE   updated_weights.shape = (3, 3, 3, 16)
RETURN     Status: Success | Type: numpy.ndarray | Size: 3.616KB


■ Computation Details ■


+-----------------------+---------+-----------------+------------------+
| Operation             |   Shape | DIM             |          CUM SUM |
|-----------------------+---------+-----------------+------------------|
| generate_RGB_matrix() |       4 | (3, 32, 32, 3)  |      1.12704e+06 |
| generate_kernels()    |       4 | (3, 3, 3, 16)   |      2.09        |
| convolution()         |       4 | (3, 30, 30, 16) |  49688.4         |
| pool()                |       4 | (3, 15, 15, 16) |  28952.2         |
| flat()                |       2 | (3, 3600)       |  28952.2         |
| dense()               |       2 | (3, 32)         | 462332           |
| dense()               |       2 | (3, 16)         |      3           |
+-----------------------+---------+-----------------+------------------+
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