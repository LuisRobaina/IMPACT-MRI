# IMPACT-MRI
Intelligent Machine-learning Powered Acceleration Technique for MRI (CS7643 Final Project)

## Description

Since September 1971, magnetic resonance imaging (MRI) scans have been helping doctors all across the world in diagnosing a myriad of diseases and injuries due to its superior capability of delivering accurate medical imaging for soft-tissue contrasts. While MRIs can produce hundreds of images from almost any direction and in any orientation, these procedures can often take up to 90 minutes in a confined space where the patient must remain still in an enclosed machine. Patients that are older or are claustrophobic may have trouble lying completely still and if they move during the scan, it may produce motion artifacts in the resulting images which the radiologists cannot use for an effective diagnosis. Additionally, images have to be acquired sequentially in k-space (an array of numbers representing spatial frequencies in the MR image) and the speed at which it collects the data is limited by hardware constraints [3]. These long scan times not only prove troublesome, but are also extremely costly; however, without a thorough examination, the machine wil not be able to deliver high-quality images. By successfully reducing the duration of MRI scan procedures without compromising the quality of the images, we can significantly increase the efficiency of hospitals and decrease the financial burden on patients, thereby optimizing the overall healthcare system.

In this project, we will study the effects of distinct algorithm design choices from both theoretical and empirical perspectives, within the bounds of our resource
constraints (compute resources, project timeline, etc). Our analysis will aid future researchers in fine-tuning self-supervised algorithms for other MRI imaging reconstruction (or related) tasks in a more efficient and expedited manner. We will combine theory and experimental data to produce a reproducible set of best-practices for this specific deep learning approach.

Similar studies have been done in other fields. For example, Andrychowicz et al. performed a similar study in the field of Reinforcement Learning. The authors identified that "while RL algorithms are often conceptually simple, their state-of-the-art implementations take numerous low- and high-level design decisions that strongly affect the performance of the resulting agents. Those choices are usually not extensively discussed in the literature, leading to discrepancy between published descriptions of algorithms and their implementations".

## Table of Contents
1. [Installation](#installation)
2. [Datasets](#datasets)
3. [Contributors](#contributors)

## Installation
We recommend the following steps to install and reproduce the results from this work:

There are two tools needed to install this project `pyenv` to manage python versions, and `poetry` to manage project dependencies.

### 1. Pyenv install
You first need to install pre-requisites for `pyenv`:

```bash
# FOR Ubuntu
sudo apt-get update
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl python3-distutils libjpeg8-dev zlib1g-dev


# FOR MacOS
brew install openssl readline sqlite3 xz zlib
```

Install `pyenv` with the official installer:
```bash
curl https://pyenv.run | bash
```

Load pyenv automatically by appending the following to `~/.bash_profile` if it exists, otherwise `~/.profile` (for login shells) and `~/.bashrc` (for interactive shells):

```bash
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

Now we can build a python environment with `pyenv`:
```bash
pyenv install 3.10.10
pyenv global 3.10.10
```

### 2. Poetry install
We will install `poetry` and then create a virtual environment using the `poetry.lock` file in the repository to make sure your environment has all the correct dependencies.

First, install `poetry` for `Mac OS / Linux` with the official installer:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Don't forget to add `poetry` to your path with
```bash
export PATH="$HOME/.local/bin:$PATH"
```

At this point all dependencies are ready to setup the project.

### IMPACT-MRI Project Setup

### 1. Set up the virtual environment with poetry:**

```bash
poetry config virtualenvs.in-project true # stores virtualenv in project directory
poetry env use 3.10.10
poetry shell
```

> Note: a .venv directory should exists inside your /IMPACT-MRI directory.
> after `poetry shell` the environment (impact-mri-py3.8) should be active.

### 2. Install dependencies in the venv
```bash
pip install -e .
poetry install
```

### 3. Test the installation
inside the venv run the following python command:

```bash
python -c "from fastmri.pl_modules import FastMriDataModule, UnetModule"
```
> If you don't see any errors then it is likely that the installation was successful.


## Datasets
Main project dataset: The Knee Portion of fastMRI Dataset (https://fastmri.med.nyu.edu/) is a good candidate dataset for this project because it was collected and approved by the leading institution NYU and passed relevant institutional reviews. Because this dataset is well established for this challenge, we wil be able to compare results with other methods that have published results for the fastMRI challenge.

**Other supporting datasets:**
More fastMRI Data: https://github.com/facebookresearch/fastMRI
Raw K-Space MRI Data: http://mridata.org/

### download the dataset
Run the following commands from the `IMPACT-MRI/` repository root directory

```bash
wget -O data/knee_singlecoil_dataset.tar.xz "https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_singlecoil_test.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=z5gwtap4eKUKoi8LMHv%2BP4Lw5mc%3D&Expires=1697853709"

cd data/
tar -xf knee_singlecoil.tar.xz
rm knee_singlecoil.tar.xz
```

## Contributors
- **Chen Yan Michael Chou**: (cchou85@gatech.edu)
- **Luis Robaina**: (lrobaina3@gatech.edu)
