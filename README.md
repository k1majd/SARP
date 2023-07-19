# Safety-Aware Repair with Predictive models
![teaser](assets/teaser4.jpg)
## Introduction
This repo includes the code for Safety-Aware Repair with Predictive models (SARP).
The proposed method combines behavioral cloning with neural network repair in a two-step supervised learning framework. 
In the first step, a policy is learned from expert demonstrations, while the second step utilizes a predictive model to impose safety constraints on the policy by repairing its predicted system properties. 
The incorporation of predictive models eliminates the need for repeated interaction with the robot, saving time on lengthy simulations typically required in reinforcement learning. 
The predictive models can encompass various aspects relevant to robot learning applications, such as proprioceptive states and collision possibilities. 
This repo showcase the implementation of SARP in two different applications: robot collision avoidance in a hospital simulation and regulation of pressure and action rate in real-world lower-leg prostheses. 

## Setup
We use the Poetry tool which is a dependency management and packaging tool in Python. It allows you to declare the libraries your project depends on and it will manage (install/update) them for you. Please follow the installation of poetry [here](https://python-poetry.org/docs/#installation). 
You can run the code in a Python virtual environment using: 

    poetry shell

To install the dependencies, run the following command:

    poetry install

## Quick Start
![quick_start](assets/quick_run_demo.gif)
Before running the examples, please download the full dataset from: [dataset link](https://drive.google.com/drive/folders/1SELQ2BnnqfwDSjb59tPyDQ6iak-kcBTm?usp=sharing). Locate each `/data` folder under the main subfolder for each example. 

### Pre-training 

Scripts for pre-training the policy and predictive models are available in each example subfolder. For the hospital showcase, start by running [0_pretrain_policy.py](examples/1_hospital_simulation/0_pretrain_policy.py)to pre-train the policy. Then, proceed with training the predictive model using [1_pretrain_predictive_model.py](examples/1_hospital_simulation/1_pretrain_predictive_model.py). 
For the prosthesis case, you should run [1_pretrain_policy_full_obs.py](examples/2_lower_leg_prosthesis/1_pretrain_policy_full_obs.py) for the fully observable case and [1_pretrain_policy_partial_obs.py](examples/2_lower_leg_prosthesis/1_pretrain_policy_partial_obs.py) for the partially observable case to pre-train the policy. 
Subsequently, use [2_pretrain_predictive_model_full_obs.py](examples/2_lower_leg_prosthesis/2_pretrain_predictive_model_full_obs.py) and [2_pretrain_predictive_model_partial_obs.py](examples/2_lower_leg_prosthesis/2_pretrain_predictive_model_partial_obs.py) to train the corresponding predictive models for each case. 
Upon completion, the trained models will be available in the `/trained_models` folder located within the main subfolder for each example.

### Repairing with SARP
Details on how to repair the policy using SARP for the hospital case is outlined in the following notbook: [2_sarp_repair.ipynb](examples/2_sarp_repair.ipynb)
For the prosthesis case, please follow the following repair notebook tutorials under `examples/2_lower_leg_prosthesis` subfolder: [3_sarp_repair_full_obs.ipynb](examples/2_lower_leg_prosthesis/3_sarp_repair_full_obs.ipynb) for the fully observable case, and [3_sarp_repair_partial_obs.ipynb](examples/2_lower_leg_prosthesis/3_sarp_repair_partial_obs.ipynb) for the partially observable case.

