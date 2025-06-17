# Traffic Sign Recognition using Convolutional Neural Networks

**A comprehensive implementation of CNN for German traffic sign classification achieving 99.65% accuracy**

---

## Overview

This project presents a sophisticated Convolutional Neural Network (CNN) implementation for classifying German traffic signs, developed as part of an Advanced Machine Learning midterm examination. The system demonstrates deep learning principles applied to computer vision, achieving exceptional performance with 99.65% accuracy on the test dataset.

The implementation showcases modern deep learning techniques including data augmentation, dropout regularization, and adaptive learning rate scheduling, all implemented using MATLAB's Deep Learning Toolbox. This project serves as a practical demonstration of how theoretical machine learning concepts translate into real-world applications with measurable performance outcomes.

## Dataset and Problem Statement

The German Traffic Sign Recognition Benchmark represents a challenging multi-class classification problem with nine distinct speed limit categories: 20, 30, 50, 60, 70, 80, 100, 120 km/h, and "End Speed Limit." The dataset contains 900 training images and 288 test images, with each class perfectly balanced at 100 training samples per category. This balanced distribution ensures that the model learns equally from all classes without inherent bias toward any particular speed limit category.

The images present several computational challenges including varying dimensions, lighting conditions, and image quality. Some samples appear significantly degraded, simulating real-world conditions where traffic signs may be partially obscured, weathered, or captured under suboptimal lighting. These challenges make the dataset particularly suitable for evaluating the robustness of deep learning architectures.

## CNN Architecture Design

The network architecture represents a carefully designed deep convolutional neural network optimized for small-scale image classification. The design philosophy balances model complexity with computational efficiency, incorporating modern regularization techniques to prevent overfitting while maintaining high representational capacity.

### Input Processing and Normalization

The network begins with an image input layer configured for 32×32 grayscale images with Z-score normalization. This preprocessing step standardizes pixel intensities to have zero mean and unit variance, significantly improving convergence stability during training. The decision to convert RGB images to grayscale reduces computational complexity while preserving the essential geometric features necessary for traffic sign recognition.

### Convolutional Feature Extraction

The feature extraction component consists of three progressive convolutional blocks, each designed to capture increasingly complex visual patterns. The first block employs two 3×3 convolutional layers with 32 filters, utilizing same-padding to preserve spatial dimensions. This configuration allows the network to learn fundamental edge detectors and simple geometric patterns while maintaining the full spatial resolution of the input.

The mathematical foundation of convolution operations can be expressed as:

```
(f * g)[n] = Σ f[m] · g[n-m]
```

Where the convolution operation slides learned kernels across the input feature maps, computing dot products to generate activation maps. The 3×3 kernel size represents an optimal balance between receptive field coverage and computational efficiency, enabling the network to capture local spatial relationships effectively.

The second convolutional block doubles the filter count to 64, enabling the network to learn more sophisticated feature combinations. This progressive increase in filter depth follows established architectural principles where early layers detect simple features while deeper layers combine these into complex representations. The third block maintains 64 filters, providing sufficient representational capacity without unnecessary computational overhead.

### Pooling and Dimensionality Reduction

Each convolutional block concludes with 2×2 max pooling operations that serve dual purposes: spatial dimensionality reduction and translation invariance. Max pooling mathematically selects the maximum activation within each 2×2 spatial window:

```
pooled[i,j] = max(input[2i:2i+1, 2j:2j+1])
```

This operation reduces computational load for subsequent layers while ensuring that the network remains robust to small spatial translations of input features.

### Regularization Strategy

Dropout layers with 25% probability follow each pooling operation, implementing a sophisticated regularization technique that randomly zeroes neural activations during training. This approach prevents co-adaptation of neurons and improves generalization by forcing the network to maintain multiple independent pathways for prediction. The mathematical effect of dropout can be modeled as:

```
output = input · mask / (1 - dropout_rate)
```

Where the mask contains Bernoulli-distributed random variables, and the scaling factor compensates for the reduced activation magnitudes.

### Classification Head

The classification component begins with a flatten layer that converts the final 4×4×64 feature maps into a 1024-dimensional vector. A fully connected layer with 512 neurons provides sufficient capacity for learning complex decision boundaries while remaining computationally tractable. The final classification layer contains exactly 9 neurons corresponding to the traffic sign categories, followed by a softmax activation that produces a probability distribution over classes.

The softmax function ensures that output probabilities sum to unity:

```
softmax(zi) = exp(zi) / Σ exp(zj)
```

This probabilistic output enables confident classification decisions and provides interpretable confidence scores for each prediction.

## Training Strategy and Optimization

The training methodology incorporates several advanced techniques designed to maximize model performance while preventing overfitting. The Adam optimizer provides adaptive learning rate adjustment for each parameter, combining the benefits of momentum-based optimization with per-parameter learning rate scaling.

### Data Augmentation

Real-time data augmentation artificially expands the training dataset through random transformations including ±15° rotations, ±2 pixel translations, and 0.9-1.1× scaling factors. These transformations improve the model's ability to recognize traffic signs under various real-world conditions without requiring additional labeled data. The augmentation strategy specifically addresses the limited training data by generating diverse variations that maintain semantic meaning while increasing visual diversity.

### Learning Rate Scheduling

The training employs a piecewise learning rate schedule beginning at 0.001 and reducing by a factor of 0.1 every 125 epochs. This approach allows aggressive learning during early training phases while enabling fine-tuned convergence as the model approaches optimal parameters. The learning rate schedule directly impacts convergence quality, with higher initial rates enabling rapid progress and lower final rates ensuring stable convergence.

### Early Stopping and Validation

Validation patience of 5 epochs implements early stopping based on validation loss, preventing overfitting while reducing unnecessary computational expense. This technique monitors validation performance and terminates training when no improvement occurs for the specified patience duration, ensuring that the model generalizes well to unseen data rather than merely memorizing training examples.

## Results and Performance Analysis

The trained model achieved exceptional performance with 99.65% accuracy on the independent test dataset, correctly classifying 287 out of 288 test images. This performance level demonstrates the effectiveness of the architectural choices and training methodology employed.

![Training Progress](assets/training_progress.png)

The training progress visualization reveals smooth convergence with minimal overfitting, indicating that the regularization strategies effectively prevented the model from memorizing training data. The validation loss closely tracks training loss throughout the training process, confirming that the model learned generalizable representations rather than dataset-specific patterns.

The confusion matrix analysis reveals that misclassifications remain extremely rare, with the single error likely attributable to ambiguous image quality or borderline cases where even human classification might prove challenging. This level of performance exceeds many published benchmarks for similar traffic sign recognition tasks.

## Technical Implementation Details

The implementation leverages MATLAB's Deep Learning Toolbox, utilizing GPU acceleration for efficient training and inference. The choice of MATLAB provides several advantages including built-in visualization tools, comprehensive deep learning functions, and seamless integration with data preprocessing pipelines.

Key implementation decisions include the use of augmented image datastores for memory-efficient data loading, automatic mixed precision training for improved performance, and systematic hyperparameter selection based on validation performance. The modular code structure facilitates experimentation with different architectural choices while maintaining reproducible results.

## Academic Recognition

![Teacher Feedback](assets/teacher_feedback.png)

The project received excellent academic recognition, achieving a score of 9.9/10 points based on the exceptional test accuracy. This recognition validates both the technical implementation quality and the theoretical understanding demonstrated throughout the project development.
