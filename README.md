## Scalable Diffusion Models with State Space Backbone （DiS）<br><sub>Official PyTorch Implementation</sub>

This repo contains PyTorch model definitions, pre-trained weights and training/sampling code for our paper exploring diffusion models with state space backbones (DiSs).


## Envs. for Pretraining 

- Python 3.10.13
  - `conda create -n your_env_name python=3.10`

- Install ``causal_conv1d`` and ``mamba``
  - `pip install -e causal-conv1d`
  - `pip install -e mamba`
