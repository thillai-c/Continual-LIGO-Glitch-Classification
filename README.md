# Continual LIGO Glitch Detection with Vision Transformer & Avalanche Framework

## Overview

This project focuses on detecting glitches in the **LIGO (Laser Interferometer Gravitational-Wave Observatory)** data streams using a **continual learning architecture**. The architecture adaptively learns from successive data streams, preserving prior knowledge and preventing **catastrophic forgetting**.

The core of the project uses a **Vision Transformer (ViT)** model alongside other architectures such as **CNN** and **Autoencoder**, all implemented with the **Avalanche Continual Learning Framework**. The continual learning strategies used in this project include **Naive** and **Replay**.

## Table of Contents

- [Architecture](#architecture)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Continual Learning Strategies](#continual-learning-strategies)
- [Sample Glitch Data Streams](#sample-glitch-data-streams)
- [References](#references)

## Architecture

![Architecture Diagram](images\continual.png)

*Figure 1: Diagram of the continual learning architecture used for LIGO glitch detection.*

## Setup and Installation

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- CUDA-enabled GPU (optional, but recommended for training)
- PyTorch 1.10+
- Avalanche Framework

### Install Dependencies

1. Clone the repository:

    ```bash
    git clone https://github.com/thillai-c/Continual-LIGO-Glitch-Classification.git
    cd continual-ligo-glitch-detection
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

The key dependencies include:
- `torch`: For building deep learning models.
- `avalanche-lib`: For continual learning support.
- `numpy`, `matplotlib`, and `pandas`: For data manipulation and visualization.
- `scikit-learn`: For additional utilities.

## Usage

1. **Open Jupyter Notebook**:
   Launch Jupyter Notebook in the project directory and open the provided notebook:

    ```bash
    jupyter notebook
    ```

2. **Run the Notebook**:
   The notebook is designed to guide you through the following steps:
   - Loading and preprocessing the LIGO dataset.
   - Implementing and training different models like Vision Transformer, CNN, and Autoencoder.
   - Applying continual learning strategies (Naive and Replay).
   - Visualizing the results and performance of the models across different data streams.

   Follow the instructions provided in the notebook to train the models using different continual learning strategies.

## Continual Learning Strategies

The following **Continual Learning** strategies are implemented in this project:

1. **Naive Strategy**:
   - A simple strategy where the model is retrained from scratch on new data without any preservation of past knowledge.

2. **Replay Strategy**:
   - A more sophisticated strategy where the model reuses a subset of old data during the training of new data, helping to reduce catastrophic forgetting.

Both strategies are implemented using the **Avalanche Continual Learning Framework**. You can read more about Avalanche [here](https://avalanche.continualai.org/).

## Sample Glitch Data Streams

![Sample Glitch Data](images\ligo_data.png)

*Figure 2: Example of glitch data streams from the LIGO dataset.*

## References

- [Avalanche Continual Learning Framework](https://avalanche.continualai.org/)
- [LIGO Scientific Collaboration](https://www.ligo.org/)
