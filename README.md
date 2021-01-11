## Study of the Variational Auto-Encoder

This repository is made as part of an assignment for the "Bayesian Learning" class of the [University of Lille's Msc. in Data Science](http://bit.ly/MasterDSULille).

#### Author : Rony Abecidan

Here, I am studying the pionneering paper of VAEs called [**"Auto-Encoding Variational Bayes"**](https://arxiv.org/abs/1312.6114) written by Diederik P. Kingma and Max Welling 

In this paper, the authors proposes a **'stochastic variational inference and learning algorithm that scales to large dataset'** which corresponds to the first VAE model. In practice, the algorithm proposed in this paper can be applied to tasks of different kinds such as denoising, impainting and super-resolution. In this work, I treat exclusively the generative purpose since it's the most famous application of VAEs and it's also the kind of task chosen by the authors for their experiments.

***

This repo is made of **3** parts :

- The article studied in a .pdf format

- A short report discussing about the strategy proposed in the paper for solving the generation problem with some additional information enable to better understand it.

- An illustrative notebook in which I propose an experiment enabling to save time in the design process of a VAE for a particular problem. This experiment is detailed in the report and the notebook.

***

## Installation

If you want to test my implementation for the MAB-VAE you'll have to install the requirements listed in requirements.txt. 

```bash
pip install -r requirements.txt
```

A part of the code is inspired by the VAEs presented in [this repo](https://github.com/AntixK/PyTorch-VAE).

If you want to see more impressive implementations of VAEs in pytorch-lightning, I advise you to check it =)
