<h1>Convolutional Neural Network Framework Project</h1>

<h1>Progress</h1>

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


METHOD     convolve(rgb_matrix: numpy.ndarray, kernels: numpy.ndarray, kernel: KernelConfig)
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


METHOD     dense(flattened_map: numpy.ndarray, neurons: int)
VARIABLE   flattend_map.shape = (3600,) | weights.shape = (64, 3600) | weighted_map.shape = (64,)
RETURN     Status: Success | Type: numpy.ndarray | Size: 0.624KB


■ Computation Details ■


CUM SUM    feature_map:   2763.535
CUM SUM    pooled_map:    2223.125
CUM SUM    flattened_map: 2223.125
CUM SUM    weighted_map:  71171.295

```

<h1>Dependencies</h1>

```bash
brew install pillow
```