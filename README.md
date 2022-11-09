# Caloscore official repository

In this repository, the implementation of the studies presented in the paper: [Score-based Generative Models for Calorimeter Shower Simulation](https://arxiv.org/abs/2206.11898).

[![DOI](https://zenodo.org/badge/501829084.svg)](https://zenodo.org/badge/latestdoi/501829084)


![The score-based generative model is trained using a diffusion process that slowly perturbs the data. Generation of new samples is carried out by reversing the diffusion process using the learned score-function, or the gradient of the data density. For different time-steps, we show the distribution of deposited energies versus generated particle energies (top) and the energy deposition in a single layer of a calorimeter (bottom), generated with our proposed CaloScore model.](./assets/caloscore_scheme.png)

[Tensorflow 2.6.0](https://www.tensorflow.org/) was used to implement all models, based on the score-model implementation from [Score-Based Generative Modeling through Stochastic Differential Equations](https://github.com/yang-song/score_sde)

# Data

Results are presented using the [Fast Calorimeter Data Challenge dataset](https://calochallenge.github.io/homepage/) and are available for download on zenodo:
* [Dataset 1](https://zenodo.org/record/6368338)
* [Dataset 2](https://zenodo.org/record/6366271)
* [Dataset 3](https://zenodo.org/record/6366324)

# Run the training scripts with

```bash
cd scripts
python train.py  --config CONFIG --model MODEL
```
* MODEL options are: subVPSDE/VESDE/VPSDE

* CONFIG options are ```[config_dataset1.json/config_dataset2.json/config_dataset3.json]```

# Sampling from the learned score function

```bash
python plot_caloscore.py  --nevts N  --sample  --config CONFIG --model MODEL
```
# Creating the plots shown in the paper

```bash
python plot_caloscore.py  --config CONFIG --model MODEL --nslices 1
```


