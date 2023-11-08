# INDUCING CAUSAL STRUCTURE PVR-MNIST

This is my (unofficial) implementation and evaluation of PVR-MNIST example on

  ```
  A. Geiger, Z. Wu, H. Lu, J. Rozner, E. Kreiss, T. Icard, N. D. Goodman, and C. Potts, “Inducing causal structure for interpretable neural networks,” 2021, Paper in arXiv.
  ```


## Table of Contents
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
  - [Reporting](#reporting-issues)
  - [Pull requests](#pull-requests)
<!-- - [Acknowledgments](#acknowledgments) -->

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.7
- Pip 22.2.2

### Installation

Provide step-by-step instructions on how to install and set up the project. Include any necessary configuration steps.

1. **Clone the repository**:

  ```sh
  git clone git@github.com:aespogom/causal_PVR_MNIST.git
  ```
2. **Navigate to the project**
  ```
  cd CAUSAL_PVR_MNIST
  ```
3. **Create a virtual environment**
4. **Install dependencies**
  ```
  pip install -r requirements.txt
  ```
5. **Run the project**
  ```
  python -m causal_PVR_MNIST.py
  ```

## Usage

The main script is placed at 
```CAUSAL_PVR_MNIST/causal_PVR_MNIST.py```. This script is meant to load the data, create the dataloaders, load the models, train and evaluate the results.

Firstly, the data is downloaded in ```CAUSAL_PVR_MNIST/data/```.

Then, the dataset model is placed at ```CAUSAL_PVR_MNIST/dataset/BlockStylePVR.py```, following the example from [OfirKedem](https://github.com/OfirKedem/Pointer-Value-Retrieval/blob/main/datasets/visual_block_style.py).

Both models, student and teacher are placed at ```CAUSAL_PVR_MNIST/models/```. 

- In the case of the student, we are using the Resnet18 architecture as the paper states. The forward and backward implementation is customized to satisfy the IIT.
- In the case of the teacher, an oracle model is implemented as a look-up table. 

The required configurations are placed at ```CAUSAL_PVR_MNIST/training_configs/```. 
- ```/MNIST.nm``` provides the required information for the interchange interventions. 
- ```/config_MNIST.yaml``` provides the required configuration for the training. 

The final weights for the student model are stored in ```CAUSAL_PVR_MNIST/results/```

## Contributing

We welcome and encourage community contributions to improve this project. To contribute, please follow these guidelines:

### Reporting Issues

If you encounter any issues or have suggestions for improvements, please check the [Issues](https://github.com/aespogom/causal_PVR_MNIST/issues) section to see if the topic has already been discussed. If not, open a new issue, including:

- A clear and descriptive title.
- A detailed description of the issue or suggestion.
- Steps to reproduce the issue if it's a bug.
- Your environment (e.g., operating system, browser, or version of the project).
- Any relevant screenshots or error messages.

### Pull Requests

We accept contributions in the form of pull requests. To submit a pull request, please follow these steps:

1. Fork the repository.
2. Create a new branch with a descriptive name for your feature or bug fix.
3. Make your changes and ensure they are well-documented.
4. Run any necessary tests and ensure they pass.
5. Commit your changes with a clear and concise commit message.
6. Push your branch to your fork.
7. Create a pull request, including a description of your changes.

<!-- ## Acknowledgments
Mention any individuals, projects, or resources that inspired or helped your project. -->


