# Project Title

## Introduction

Training Models to generate 3D point clouds from 2D images using Few-shot learning. This repo comprises of many experiments and models. The models are trained using the ShapeNet dataset.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages.

```sh
pip install -r requirements.txt
```

## Usage

To train the model, run the following command:

```sh
python run.py
```

## Modules

### Extensions

This directory contains the following sub-modules:

Chamfer Distance: A module for calculating the Chamfer distance. Implemented in chamfer_cuda.cpp and chamfer.cu.

## Models

This directory contains the models used in the project, including:

- GAN: Implemented in [gan.py](models/gan.py).
- PCN: Implemented in [pcn.py](models/pcn.py).

## Utils

This directory contains utility functions used across the project, implemented in [utils.py](utils/utils.py).
