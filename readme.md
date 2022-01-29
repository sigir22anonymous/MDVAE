# Supplementary Material: Multimodal Disentanglement Variational AutoEncoders with Counter-Intuitive Cross-Reconstruction for Zero-Shot Cross-Modal Retrieval

This repository is the Pytorch implementation of our MDVAE method.



## Installation and Requirements

- ```
  cudatoolkit=11.1.1
  ```

- ```
  numpy=1.20.3
  ```

- ```
  python=3.9.6
  ```

- ```
  pytorch=1.8.0
  ```




## Codes

**config.py** is the script used to set hyper-parameters for constructing model and training.

**dataloader.py** is the script used to load data for training and testing.

**main_sketch.py** is the the script used to train and test MDVAE.

**model.py** is the script that contains implementation of the MDVAE model.

**utils.py** is the script that contains some implementation of initialization, losses, and training function.