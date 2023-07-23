# IMPACT-MRI
Intelligent Machine-learning Powered Acceleration Technique for MRI
(CS7643 Final Project)

## Description

Since September 1971, magnetic resonance imaging (MRI) scans have been helping doctors all across the world in diagnosing a myriad of diseases and injuries due to its superior capability of delivering accurate medical imaging for soft-tissue contrasts. While MRIs can produce hundreds of images from almost any direction and in any orientation, these procedures can often take up to 90 minutes in a confined space where the patient must remain still in an enclosed machine. Patients that are older or are claustrophobic may have trouble lying completely still and fi they move during the scan, ti may produce motion artifacts in the resulting images which the radiologists cannot use for an effective diagnosis. Additionally, images have to be acquired sequentially in k-space (an array of numbers representing spatial frequencies in the MR image) and the speed at which it collects the data is limited by hardware constraints [3]. These long scan times not only prove troublesome, but are also extremely costly; however, without a thorough examination, the machine wil not be able to deliver high-quality images. By successfully reducing the duration of MRI scan procedures without compromising the quality of the images, we can significantly increase the efficiency of hospitals and decrease the financial burden on patients, thereby optimizing the overall healthcare system.

In this project, we will study the effects of distinct algorithm design choices from both theoretical and empirical perspectives, within the bounds of our resource
constraints (compute resources, project timeline, etc). Our analysis will aid future researchers in fine-tuning self-supervised algorithms for other MRI imaging reconstruction (or related) tasks in a more efficient and expedited manner. We will combine theory and experimental data to produce a reproducible set of best-practices for this specific deep learning approach.

Similar studies have been done in other fields. For example, Andrychowicz et al. performed a similar study in the field of Reinforcement Learning. The authors identified that "while RL algorithms are often conceptually simple, their state-of-the-art implementations take numerous low- and high-level design decisions that strongly affect the performance of the resulting agents. Those choices are usually not extensively discussed in the literature, leading to discrepancy between published descriptions of algorithms and their implementations".

## Table of Contents
1. [Installation](#installation)
2. [Data](#data)
3. [Usage](#usage)
4. [Results](#results)
5. [Contributors](#contributors)

## Installation

### Dependencies
This project requires the following dependencies:

- Python (>= 3.6)
- Numpy (>= 1.16.0)
- Scipy (>= 1.4.1)
- Sklearn (>= 0.22.1)

To install these dependencies, you can use pip:

```bash
pip install -r requirements.txt
```

## Contributors
- **Chen Yan Michael Chou**: (cchou85@gatech.edu)
- **Luis Robaina**: (lrobaina3@gatech.edu)


