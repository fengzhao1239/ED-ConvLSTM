# An encoder-decoder ConvLSTM surrogate model for simulating geological CO2 sequestration with dynamic well controls


This study delves into the application of the **Encoder-Decoder Convolutional Long Short-Term Memory (ED-ConvLSTM)** neural network for predicting the complex evolution of state variables under dynamic CO2 injection schemes.    
[Inception](https://arxiv.org/pdf/1409.4842) blocks enhanced with light-weighted attention modules are introduced in the encoder to extract high-dimensional input features. [ConvLSTM](https://proceedings.neurips.cc/paper_files/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf) is employed to propagate spatial temporal information in the low-dimensional latent space. Further, progressive upsampling blocks are used to reconstruct the latent features for the desired output. Instead of taking discrete time steps as an input feature, the proposed network captures the dynamic dependencies with the inherent ConvLSTM cell. The network has access to data at only *portion of the initial time steps* during training stage, while it is used to predict the state variables at ***unseen time steps*** during testing stage.   
Results show that the network can produce excellent predictions for both pressure and saturation, even at unseen future time steps. The remarkable generalizability to different geological permeability fields is also evaluated. ED-ConvLSTM outperforms the standard [U-Net](https://arxiv.org/pdf/1505.04597) by far, especially when predicting beyond the training time period. These numerical experiments demonstrate the advantages of ED-ConvLSTM in terms of prediction accuracy, extrapolability and generalizability.

![image](https://github.com/fengzhao1239/ED-ConvLSTM/blob/main/workflow.jpg)

## File list:
This repository contains all the codes for the training of **ED-ConvLSTM** and the baseline **U-Net**.  
- Train: the training script for ED-ConvLSTM
- Model_simple: the model ED-ConvLSTM
- MyDataset: the dataset reader for ED-ConvLSTM
- Unet_train: the training script for U-Net
- Unet_model: the model U-Net
- Unet_dataset: the dataset reader for U-Net
- all the plot_xxx.ipynb: plotting jupyternotebooks
- model weights for pressure and saturation are also included (.pth file)

*Note:*

The training/validation/testing dataset has been uploadded to google drive: https://drive.google.com/file/d/1zCInuI41NOmJV_BBICABCiH4pwz7IEs2/view?usp=sharing

The extrapolation and generalization dataset is not included.

## Paper
The published paper for this repository can be found:

Feng, Zhao, et al. "An encoder-decoder ConvLSTM surrogate model for simulating geological CO2 sequestration with dynamic well controls." Gas Science and Engineering (2024): 205314.

https://doi.org/10.1016/j.jgsce.2024.205314
