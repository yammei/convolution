import numpy as np
from method_logger import ML, log
from img_translation import *
from tabulate import tabulate
import matplotlib.pyplot as plt

class KernelConfig:
    def __init__(self):
        self.weight: float = .1
        self.size: int = 3
        self.num: int = 16
        self.stride: int = 1

class TrainingConfig:
    def __init__(self, labels: np.ndarray):
        # Reference: Gervais N. (2020). How many neurons should be in the last layer of the neural network? https://stackoverflow.com/a/63381557
        self.labels: np.ndarray = labels # Needs to have same # of class options as neurons in final layer.

class Activation:
    def __init__(self):
        pass

    # ReLU removes all negative value, which helps introduce non-linearity and reduce less useful learned features causing slow learning.
    # Reference: H. Morten. (2023). Convolutional (CNN) and Recurrent (RNN) Neural Networks. https://compphysics.github.io/MLErasmus/doc/pub/day7/html/day7.html#:~:text=This%20is%20then%20passed%20through,in%20the%20first%20convolutional%20layer
    # Reference: Deleted User (2022). Why is it desirable to suppress or remove the negative portion of activation functions? What is the theoretical explanation or intuition behind it?. r/MLQuestions. https://www.reddit.com/r/MLQuestions/comments/twtou9/comment/i3hhcmf/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button
    def relu(x: np.ndarray) -> np.ndarray:
        log(f"\n┌----┐")
        ML.start('relu', {'x': type(x)})

        # Removes all values below 0. Effectively all negative values.
        activated_map = np.maximum(0, x)

        ML.end(1, activated_map)
        log(f"└----┘\n")
        return activated_map

    # Softmax is good for multi-classification models. Cummulative sum of all probabilities will be 1.
    # C. Bala Priya. (2023). Softmax Activation Function: Everything You Need to Know. ML Foundations. https://www.pinecone.io/learn/softmax-activation/
    # S. Josh (2021). Neural Networks Part 5: ArgMax and SoftMax. https://youtu.be/KpKog-L9veg?si=d988QkT6BkOC_Lep
    def softmax(x: np.ndarray) -> np.ndarray:
        log(f"\n┌----┐")
        ML.start('softmax', {'x': type(x)})

        def plot(probabilities: np.ndarray) -> np.ndarray:
            plt.plot(probabilities, marker='o', linestyle='-', color='b')
            plt.title('Softmax Probabilities')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.show()

        if x.ndim == 1:
            # Element-wise subtraction of max number on all elements in input array and exponentation of e^x. x being all elements in the array.
            exp_x = np.exp(x - np.max(x))
            # e^x / SUM(e^x_0 + ... + e^x_n-1)
            probabilities = exp_x / np.sum(exp_x)
            log(f"VARIABLE   sorted(probabilities[0:5]) = {sorted(probabilities[0:5])}")
            # plot(probabilities)
        elif x.ndim > 1:
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            probabilities = exp_x / np.sum(exp_x, axis=1, keepdims=True)
            # plot(probabilities)

        ML.end(1, probabilities)
        log(f"└----┘\n")
        return probabilities

    # Will be good for binary classification for the final layer's activation function, and leading into binary cross-entropy.
    # Note: Explore (T. Chang-Hung at al., 2015) for later implementation.
    # Reference: T. Chang-Hung et al. (2015). A Hardware-Efficient Sigmoid Function With Adjustable Precision for a Neural Network System. IEEE Transactions on Circuits and Systems II: Express Briefs. https://ieeexplore.ieee.org/abstract/document/7159035
    # Reference: H. Antti. (2012). Fast Sigmoid Algorithm. https://stackoverflow.com/a/10733861
    def sigmoid(x: np.ndarray) -> np.ndarray:
        ML.start('sigmoid', {'x': type(x)})

        # Translates values into probable ranges from 0 to 1. Larger numbers (x) result closer to 1, smaller numbers (x) result closer to 0.
        # 1 / (1 + e^-x)
        probabilities: np.ndarray = 1 / (1 + np.exp(-x))

        ML.end(1, probabilities)
        pass

class Computation:
    # 2D Matrix * 2D Matrix -> 2D Matrix
    @staticmethod
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

    # 3D Kernel * 1D Number of Kernels -> 4D Kernels
    # Reference: S. Sabyasachi. (2018) Deciding optimal kernel size for CNN. https://towardsdatascience.com/deciding-optimal-filter-size-for-cnns-d6f7b56f9363
    @staticmethod
    def generate_kernels(kernel: dict = KernelConfig()) -> np.ndarray:
        ML.start(func_name='generate_kernels', args={'kernel_weight': type(kernel.weight), 'kernel_size': type(kernel.size), 'kernel_num': type(kernel.num)})

        # Generates kernel.num amount of kernels 3x3 kernels each with 3 channels for RGB, or simply 3x3x3x16.
        kernels: np.ndarray = np.random.randn(kernel.size, kernel.size, 3, kernel.num) * kernel.weight
        log(f"VARIABLE   kernels.shape = {kernels.shape}")

        ML.end(status=1, return_val=kernels)
        return kernels

    # 3D RGB Matrix * 3D Kernel Slice -> 2D Feature Map * 1D Num of Kernel Slices -> 3D Feature Maps
    @staticmethod
    def convolution(rgb_matrix: np.ndarray, kernels: np.ndarray, kernel: dict = KernelConfig()) -> np.ndarray:
        ML.start(func_name='convolution', args={'rgb_matrix': type(rgb_matrix), 'kernels': type(kernels), 'kernel': type(kernel)})

        # Sets feature map metadata.
        batches: int = batch_rgb_matrix.shape[0]
        input_height, input_width, _ = batch_rgb_matrix.shape[1:]
        output_height: int = (input_height - kernel.size) // kernel.stride + 1
        output_width: int = (input_width - kernel.size) // kernel.stride + 1
        batch_feature_map: np.ndarray = np.zeros((batches, output_height, output_width, kernel.num))

        for b in range(batches):
            rgb_matrix = batch_rgb_matrix[b]
            feature_map: np.ndarray = np.zeros((output_height, output_width, kernel.num))

            for k in range(kernel.num):
                for i in range(0, input_height - kernel.size, kernel.stride):
                    for j in range(0, input_height - kernel.size, kernel.stride):
                        # Convolution operation fills out feature map width the product of rgb_matrix and kernels @ index k.
                        feature_map[i//kernel.stride, j//kernel.stride] = np.sum(rgb_matrix[i:i+kernel.size, j:j+kernel.size, :] * kernels[:, :, :, k])

            # Append results
            batch_feature_map[b] = feature_map

        # ReLU. Removes negative values.
        batch_feature_map = np.maximum(0, batch_feature_map)

        log(f"VARIABLE   batch_feature_map.shape = {batch_feature_map.shape}")
        ML.end(1, batch_feature_map)
        return batch_feature_map

    # 3D Feature Map * 3D Downsampler (2D Window * Num of Features) -> 3D Pooled Map
    # Reference: D. Matthew. (2017). Feature extracted by max pooling vs mean pooling. https://stats.stackexchange.com/questions/291451/feature-extracted-by-max-pooling-vs-mean-pooling
    @staticmethod
    def pool(batch_feature_map: np.ndarray, pool_size: int = 2, pool_stride: int = 2, pool_mode: str = 'max', kernel: dict = KernelConfig()) -> np.ndarray:
        ML.start('pool', {'batch_feature_map': type(batch_feature_map), 'pool_size': type(pool_size), 'pool_mode': type(pool_mode)})

        # Sets pooled map metadata.
        batches, input_height, input_width, _ = batch_feature_map.shape
        output_height: int = (input_height - pool_size) // pool_stride + 1
        output_width: int = (input_width - pool_size) // pool_stride + 1
        pooled_map: np.ndarray = np.zeros((batches, output_height, output_width, kernel.num))

        for b in range(batches):
            for k in range(kernel.num):
                for i in range(0, input_height - pool_size + 1, pool_stride):
                    for j in range(0, input_width - pool_size + 1, pool_stride):
                        # Downsamples feature map by taking max value in a (pool_size x pool_size) sliding window for all feature maps (k).
                        pooled_map[b, i//pool_stride, j//pool_stride, k] = np.max(batch_feature_map[b, i:i+pool_size, j:j+pool_size, k])

        log(f"VARIABLE   pooled_map.shape = {pooled_map.shape}")
        ML.end(1, pooled_map)
        return pooled_map

    # 3D Pooled Map -> 1D Flattened Map
    # Flattens multi-dimensional array to 1D array to translate to full connective (dense) layers; providing the number of input features.
    # (e.g., for a 3D array of size (15, 15, 16) you have 15 * 15 * 16 or 3600 input features)
    @staticmethod
    def flat(pooled_map: np.ndarray) -> np.ndarray:
        ML.start('flat', {'pooled_map': type(pooled_map)})

        batches = pooled_map.shape[0]
        flattened_map: np.ndarray = pooled_map.reshape(batches, -1)
        log(f"VARIABLE   flattened_map.shape = {flattened_map.shape}")

        ML.end(1, flattened_map)
        return flattened_map

    # 1D Flattened Map * 2D (Weights * Num of Neurons) -> 2D Weighted Features Map -> 1D Sum of Weighted Features Map
    # Reference: M. Stefano. et al. (2021). Effects of hidden layer sizing on CNN fine-tuning. Future Generation Computer Systems. https://www.sciencedirect.com/science/article/abs/pii/S0167739X2033082X
    # Reference: K. Sandhya. (2021). How do determine the number of layers and neurons in the hidden layer?. Geek Culture. https://medium.com/geekculture/introduction-to-neural-network-2f8b8221fbd3
    @staticmethod
    def dense(flattened_map: np.ndarray, neurons: int = 64, activation: str ='none') -> np.ndarray:
        log(f"┌------- START -------┐")
        ML.start('dense', {'flattened_map': type(flattened_map), 'neurons': type(neurons), 'activation': activation})

        batches: int = flattened_map.shape[0]
        features: int = flattened_map.shape[1]
        weights: np.ndarray = np.random.rand(neurons, features)
        biases: np.ndarray = np.random.rand(neurons)

        # Dot product operation. Creates n-amount of neurons with weighed then biased input features.
        # SUM from j=0 to n=neurons of WEIGHT_ij * INPUT_i
        weighted_map: np.ndarray = np.dot(flattened_map, weights.T) + biases
        log(f"VARIABLE   flattend_map.shape = {flattened_map.shape} | weights.shape = {weights.shape} | weighted_map.shape = {weighted_map.shape}")

        if activation == 'relu':
            activated_output = Activation.relu(weighted_map)
        elif activation == 'softmax':
            activated_output = Activation.softmax(weighted_map)
        else:
            activated_output = weighted_map

        ML.end(1, activated_output)
        log(f"└-------- END --------┘\n")
        return activated_output, weighted_map, weights, biases

    # Calculates loss values between the model output values and ground truth labels.
    # Reference: F. Brendan. (2024). Machine Learning Glossary. https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy
    # Reference: R. Poojary and A. Pai. (2019). Comparative Study of Model Optimization Techniques in Fine-Tuned CNN Models. 2019 International Conference on Electrical and Computing Technologies and Applications (ICECTA). https://ieeexplore.ieee.org/abstract/document/8959681
    @staticmethod
    def cross_entropy(true_labels: list, predicted_labels: list, mode: str = 'binary') -> float:
        predicted_labels = np.clip(predicted_labels, 1e-15, 1e-15) # Prevent log(0).
        # Positive average value of all positive and negative losses.
        loss = -np.mean(true_labels * np.log(predicted_labels) +  (1 - true_labels) * np.log(1 - predicted_labels))
        return loss

    # Updates gradient weights and biases based on loss values.
    @staticmethod
    def back_propagation(predicted_output: np.ndarray, labels: np.ndarray, previous_output: np.ndarray, weights: np.ndarray, activation: str) -> tuple:
        ML.start('back_propagation', {'predicted_output': type(predicted_output), 'labels': type(labels), 'weights': type(weights), 'activation': activation})

        if activation == 'softmax':
            gradient_output = predicted_output - labels
        elif activation == 'relu':
            gradient_output = predicted_output * (predicted_output > 0).astype(float)
        log(f"VARIABLE   predicted_output.shape = {predicted_output.shape} | gradient_output.shape = {gradient_output.shape}")

        gradient_weights = np.dot(gradient_output.T, previous_output)
        gradient_biases = np.sum(gradient_output, axis=0, keepdims=True)
        gradient_input = np.dot(gradient_output, weights)

        log(f"VARIABLE   gradient_input.shape = {gradient_input.shape} | gradient_weights.shape = {gradient_weights.shape} | gradient_biases.shape = {gradient_biases.shape}")

        ML.end(1, (gradient_input, gradient_weights, gradient_biases))
        return gradient_input, gradient_weights, gradient_biases

    @staticmethod
    def back_propagation_conv(predicted_output: np.ndarray, labels: np.ndarray, kernels: np.ndarray, kernel: KernelConfig) -> tuple:
        ML.start('back_propagation_conv', {'predicted_output': type(predicted_output), 'labels': type(labels), 'kernels': type(kernels)})

        gradient_output: np.ndarray = predicted_output - labels
        batches, input_height, input_width, kernel_num = predicted_output.shape
        derivative_kernels: np.ndarray = np.zeros_like(kernels)
        derivative_predicted_output: np.ndarray = np.zeros_like(predicted_output)

        for b in range(batches):
            for k in range(kernel_num):
                for i in range(0, input_height - kernel.size + 1, kernel.stride):
                    for j in range(0, input_width - kernel.size + 1, kernel.stride):
                        for c in range(kernels.shape[2]):
                            derivative_kernels[:, :, c, k] += np.sum(predicted_output[b, i:i+kernel.size, j:j+kernel.size, c] * gradient_output[b, i//kernel.stride, j//kernel.stride, k])
                            derivative_predicted_output[b, i:i+kernel.size, j:j+kernel.size, c] += kernels[:, :, c, k] * gradient_output[b, i//kernel.stride, j//kernel.stride, k]

        log(f"VARIABLE   derivative_predicted_output.shape = {derivative_predicted_output.shape} | derivative_kernels.shape = {derivative_kernels.shape}")

        ML.end(1, (derivative_predicted_output, derivative_kernels))
        return derivative_predicted_output, derivative_kernels

    @staticmethod
    def update_weights_and_biases(weights: np.ndarray, biases: np.ndarray, gradient_weights: np.ndarray, gradient_biases: np.ndarray, learning_rate: float = 0.001) -> tuple:
        ML.start('update_weights_and_biases', {'weights': type(weights), 'biases': type(biases), 'gradient_weights': type(gradient_weights), 'gradient_biases': type(gradient_biases), 'learning_rate': learning_rate})

        updated_weights = weights - (learning_rate * gradient_weights)
        updated_biases = biases - (learning_rate * gradient_biases)

        log(f"VARIABLE   updated_weights.shape = {updated_weights.shape} | updated_biases.shape = {updated_biases.shape}")

        ML.end(1, (updated_weights, updated_biases))
        return updated_weights, updated_biases

    @staticmethod
    def updated_kernels(currnels: np.ndarray, gradient_kernels: np.ndarray, learning_rate: float = 0.001) -> tuple:
        ML.start('updated_kernels', {'currnels': type(currnels), 'gradient_kernels': type(gradient_kernels), 'learning_rate': learning_rate})

        updated_kernels = batch_default_kernels - (0.001 * gradient_kernels)

        log(f"VARIABLE   updated_weights.shape = {updated_kernels.shape}")

        ML.end(1, updated_kernels)
        return updated_kernels

    # Parent method for training process.
    @staticmethod
    def fit(model: any, training_data: any, training_labels: np.array, validation_data: any, batch: int, epochs: int, optimizer: str = 'ADAM') -> any:
        # Load in data.
        # Compute input data through model epoch number of times.
        # Save final optimized weights and biases.
        pass

# TEST SECTION BELOW

'''
# Matrix Multiplication Test Script
np.random.seed(1)
rand_matrix_1 = np.random.randint(50, 201, size=(6, 7))
rand_matrix_2 = np.random.randint(50, 201, size=(7, 9))
multiply_matrices(matrix_1=rand_matrix_1, matrix_2=rand_matrix_2)
'''

np.random.seed(1)
img_paths: list = ['../images/training/img1.png', '../images/training/img2.png', '../images/training/img3.png']

# Single-Label Encoding: 1 = isCat, 0 = notCat. Respective to dataset sample order.
sle_labels: np.ndarray = np.array([1, 0, 1])
num_classes: int = 16
sle_labels = np.eye(num_classes)[sle_labels]
training_config = TrainingConfig(labels=sle_labels)
print(f'Training Config: {training_config.labels}')

# Forward Pass
batch_rgb_matrix: np.array = np.array([generate_RGB_matrix(img_path) for img_path in img_paths])
ML.start('generate_RGB_matrix', {})
log(f"VARIABLE   pooled_map.shape = {batch_rgb_matrix.shape}")
ML.end(1, batch_rgb_matrix)

batch_default_kernels: np.ndarray = Computation.generate_kernels()
batch_feature_map: np.ndarray = Computation.convolution(batch_rgb_matrix, batch_default_kernels)
batch_pooled_map: np.ndarray = Computation.pool(batch_feature_map)
flattened_map: np.ndarray = Computation.flat(batch_pooled_map)
activated_weighted_map_1, weighted_map_1, weights_1, biases_1 = Computation.dense(flattened_map, neurons=32, activation='relu')
activated_weighted_map_2, weighted_map_2, weights_2, biases_2 = Computation.dense(activated_weighted_map_1, neurons=16, activation='softmax')

# Backwards Pass

# 1. Backpropagation through second dense layer (softmax)
gradient_input_2, gradient_weights_2, gradient_biases_2 = Computation.back_propagation(
    predicted_output=activated_weighted_map_2,
    labels=training_config.labels,
    previous_output=activated_weighted_map_1,
    weights=weights_2,
    activation='softmax'
)
updated_weights_2, updated_biases_2 = Computation.update_weights_and_biases(
    weights=weights_2,
    biases=biases_2,
    gradient_weights=gradient_weights_2,
    gradient_biases=gradient_biases_2,
    learning_rate=.001
)

# 2. Backpropagation through first dense layer (ReLU)
gradient_input_1, gradient_weights_1, gradient_biases_1 = Computation.back_propagation(
    predicted_output=gradient_input_2,
    labels=gradient_input_2,
    previous_output=flattened_map,
    weights=weights_1,
    activation='relu'
)
updated_weights_1, updated_biases_1 = Computation.update_weights_and_biases(
    weights=weights_1,
    biases=biases_1,
    gradient_weights=gradient_weights_1,
    gradient_biases=gradient_biases_1,
    learning_rate=.001
)

# 3. Backpropagation through convolutional layer
_, gradient_kernels = Computation.back_propagation_conv(
    predicted_output=batch_feature_map,
    labels=batch_feature_map,
    kernels=batch_default_kernels,
    kernel=KernelConfig()
)
# Update the convolutional kernels
updated_kernels = Computation.updated_kernels(currnels=batch_default_kernels, gradient_kernels=gradient_kernels, learning_rate=.001)

def log_details() -> None:
    log(f"\n■ Computation Details ■\n\n")
    headers: list = ['Operation', 'Shape', 'DIM', 'CUM SUM']
    computation_details = [
        [
            "generate_RGB_matrix()",
            f"{len(batch_rgb_matrix.shape)}",
            f"{batch_rgb_matrix.shape}",
            f"{np.sum(batch_rgb_matrix):.2f}"
        ],
        [
            "generate_kernels()",
            f"{len(batch_default_kernels.shape)}",
            f"{batch_default_kernels.shape}",
            f"{np.sum(batch_default_kernels):.2f}"
        ],
        [
            "convolution()",
            f"{len(batch_feature_map.shape)}",
            f"{batch_feature_map.shape}",
            f"{np.sum(batch_feature_map):.2f}"
        ],
        [
            "pool()",
            f"{len(batch_pooled_map.shape)}",
            f"{batch_pooled_map.shape}",
            f"{np.sum(batch_pooled_map):.2f}"
        ],
        [
            "flat()",
            f"{len(flattened_map.shape)}",
            f"{flattened_map.shape}",
            f"{np.sum(flattened_map):.2f}"
        ],
        [
            "dense()",
            f"{len(activated_weighted_map_1.shape)}",
            f"{activated_weighted_map_1.shape}",
            f"{np.sum(activated_weighted_map_1):.2f}"
        ],
        [
            "dense()",
            f"{len(activated_weighted_map_2.shape)}",
            f"{activated_weighted_map_2.shape}",
            f"{np.sum(activated_weighted_map_2):.2f}"
        ]
    ]
    log(tabulate(computation_details, headers=headers, tablefmt='psql'))
    log(f"\n")

log_details()


# Resources
# L. Yongxin et al. (2023). Zero-Bias Deep Learning for Accurate Identification of Internet-of-Things (IoT) Devices. IEEE Internet of Things Journal. https://ieeexplore.ieee.org/document/9173537