# Swapping-Autoencoder [1]
A simplified version of the Swapping Autoencoder without the use of Visdom for the control of training and prepared for the training of the model on FFHQ dataset.

<img src="https://github.com/DaniMe98/Swapping-Autoencoder/blob/main/imgs/overview.jpg" width="600" />

<b> Swapping Autoencoder consists of Autoencoding (top) and swapping (bottom) of images. </b>

  <b>• (Top) </b> An encoder E embed an input <b>(Notre-Dame)</b> in the latent space. 
The decoding with generator G should produce realistic images (encouraged by a discriminator D) corresponding to the input(reconstruction loss).

  <b>• (Bottom) </b> The decoding with another image texture <b>(Saint Basil’s Cathedral)</b> should seem realistic (for D) 
Carry the image texture, by training a patch co-occurrence discriminator Dpatch that encourages the output patches to look indistinguishable from the starting image.

## Installation / Requirements

- CUDA 10.1 or newer 
- PyTorch 1.7.1 on Python 3.6
- Install dependencies with `pip install dominate torchgeometry func-timeout tqdm matplotlib opencv_python numpy GPUtil Pillow scikit-learn ninja`

The proposed method can be used to embed in an efficient way an input image in <b>factored latent space </b>to generate hybrid images by exchange of the latent codes.
Every codes in the given representation should be individually modified and the resulting image should seem realistic and show the non-modified code invariant.  

## Quick usage

To have a quicker demostration of how the model works, you can refer to the 'Simple-Swap&Inter.ipynb'.

## Training

In this project the model wil be trained/tested with different settings.
In particular, these settings were tested:
Training of a model with original structure, but in two different version:

  <b>1. </b> Version trained on the entire dataset with images at resolution <b>256x256 </b>
  
  <b>2.	</b> Version trained on half dataset at higher resolution <b>512x512 </b>
  
Training of the model as in the original work, but with some modification at the main net structure, weakened in order to have a quicker training.

The four of them will then be evaluated as in the original work, both visually and with automated metrics.

## Evaluation of the model

<b>1. Visually</b>

<img src="https://github.com/DaniMe98/Swapping-Autoencoder/blob/main/imgs/Immagine%202022-07-21%20184054.jpg" width="600" />

<b>2. FID</b> Fréchet Inception Distance computed between the dataset images and a set of generated images using https://github.com/mseitzer/pytorch-fid.

<b>3. SIFID </b> Single-Image FID [2] measure style similarity by computing Fréchet Inception Distance (FID) between two feature distributions, each generated from a single image.

## References

<b>[1] </b>Park, Taesung and Zhu, Jun-Yan and Wang, Oliver and Lu, Jingwan and Shechtman, Eli and Efros, Alexei A. and Zhang, Richard : Swapping Autoencoder for Deep Image Manipulation

<b>[2] </b>Shaham, T.R., Dekel, T., Michaeli, T.: Singan: Learning a generative model from a single natural image. 

<b>Original code: </b> https://github.com/taesungp/swapping-autoencoder-pytorch

<b>SIFID: </b> https://github.com/tamarott/SinGAN/tree/master/SIFID

