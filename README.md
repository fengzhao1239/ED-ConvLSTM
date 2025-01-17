# ED-ConvLSTM
This repository contains all the codes for the training of **ED-ConvLSTM** and the baseline **U-Net**.

## File list:
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
