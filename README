````markdown
# DCGAN From Scratch

A clean PyTorch implementation of a **Deep Convolutional Generative Adversarial Network (DCGAN)** built entirely from scratch for handwritten digit generation on the MNIST dataset. This project implements both the Generator and Discriminator architectures without relying on high-level GAN libraries, providing a practical understanding of adversarial learning and deep generative models.

---

## Features

- Deep Convolutional GAN (DCGAN) implementation from scratch
- Custom Generator using Transposed Convolution layers
- Convolutional Discriminator for real/fake image classification
- Trained on the MNIST handwritten digits dataset
- Batch Normalization for improved training stability
- CUDA/GPU acceleration support
- Modular and easy-to-understand PyTorch codebase

---

## Architecture

### Generator
- 100-dimensional latent noise vector
- Fully Connected projection layer
- ConvTranspose2D layers for progressive upsampling
- Batch Normalization
- ReLU activations
- Sigmoid output activation for grayscale image generation

### Discriminator
- Convolutional Neural Network (CNN)
- Conv2D layers for hierarchical feature extraction
- Batch Normalization
- ReLU activations
- Fully Connected output layer for binary classification

---

## Tech Stack

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib

---

## Project Structure

```text
.
├── DC_gen.py               # Generator architecture
├── DC_disc.py              # Discriminator architecture
├── train.py                # GAN training pipeline
├── mnist_dataloader.py     # Custom MNIST dataloader
├── train_images/
├── train_labels.csv
└── README.md
````

---

## Training Pipeline

1. Sample a random 100-dimensional latent vector.
2. Generate synthetic handwritten digits using the Generator.
3. Train the Discriminator on real and generated images.
4. Update the Generator using adversarial loss.
5. Repeat until the Generator produces realistic MNIST digits.

---

## Concepts Implemented

* Generative Adversarial Networks (GANs)
* Deep Convolutional GANs (DCGAN)
* Adversarial Training
* Latent Space Sampling
* Transposed Convolutions
* Convolutional Neural Networks (CNNs)
* Batch Normalization
* Binary Cross-Entropy Loss
* Image Synthesis

---

## Future Improvements

* Wasserstein GAN (WGAN)
* Conditional GAN (cGAN)
* Spectral Normalization
* Gradient Penalty
* Model Checkpointing
* TensorBoard Logging
* Higher-resolution image generation

---

## License

This project is developed for educational purposes to understand and implement Deep Convolutional Generative Adversarial Networks from scratch using PyTorch.

```
```
