# Human as AI mentor-based deep reinforcement learning (HAIM-DRL)

This repo is the implementation of the paper "Human as AI Mentor: Enhanced Human-in-the-loop Reinforcement Learning for Safe and Efficient Autonomous Driving (Accepted by Communications in Transportation Research)".


[**Webpage**](https://zilin-huang.github.io/HAIM-DRL-website/) | 
[**Code**](https://github.com/zilin-huang/HAIM-DRL) | 
[**Video**](https://www.youtube.com/playlist?list=PL-EmC8vF-RSH2j09uxZeyiVV_ePuA3Eud) |
[**Paper**](https://arxiv.org/abs/2401.03160) 


<div align=center><img src=./docs/HAIM-DRL_Poster.png ></div>



## Installation

```bash
# Clone the code to local
git clone https://github.com/zilin-huang/HAIM-DRL.git
cd HAIM-DRL

# Create virtual environment
conda create -n HAIM-DRL python=3.7
conda activate HAIM-DRL

# Install basic dependency
pip install -e .

conda install cudatoolkit=11.0
conda install -c nvidia cudnn
# Now you can run the training script of HAIM-DRL in MetaDrive Environment.
```


## Training HAIM-DRL
HAIM-DRL is designed for teaching AI to learn a generalizable autonomous driving policy efficiently and safely.
Supported by [MetaDrive](https://github.com/decisionforce/metadrive), the concrete goal of driving tasks is to drive the vehicle to the destination with as fewer collision as possible. 
Also, to prevent driving out of the road which will terminate the episode, please follow the yellow checkpoints indicating navigation information when you are training HAIM-DRL.

**Note:** we mask the reward signal for HAIM-DRL agent, so it is a reward-free method.


### Quick Start
Since the main experiment of HAIM-DRL takes one hour and requires a steering wheel (Logitech G920), we further provide an 
easy task for users to experience HAIM-DRL.
```bash
cd haim_drl/run_main_exp/
python train_haim_drl_keyboard_easy.py --num-gpus=1
```
In this task, human is authorized to take over the vehicle by pressing **W/A/S/D** and guide or safeguard the agent to 
the destination ("E" can be used to pause simulation). 
Since there is only one map in this task, 10 minutes or 5000 transitions is enough for HAIM-DRL agent to learn a policy.


### Main Experiment
To reproduce the main experiment reported in paper, run following scripts:
```bash
python train_haim_drl.py --num-gpus=1
```
If steering wheel is not available, set ```controller="keyboard"``` in the script to train HAIM-DRL agent. After launching this script,
one hour is required for human to assist HAIM-DRL agent to learn a generalizable driving policy by training in 50 different maps.


### CARLA Experimennt
CARLA used in our experiment is version 0.9.9.4, so pleas follow the instruction in 
[CARLA offical repo](https://github.com/carla-simulator/carla) to install it.
After installation, launch CARLA server by:
```bash
./CarlaUE4.sh -carla-rpc-port=9000 
```

For the interacting with CARLA core, we utilize the CARLA client wrapper implemented in [DI-Drive](https://github.com/opendilab/DI-drive), so new dependencies
is needed. We recommend initializing a **new** conda environment by:
```bash
# Create new virtual environment
conda create -n HAIM-DRL-carla python=3.7
conda activate HAIM-DRL-carla

# install DI-Engine
pip install di-engine==0.2.0 markupsafe==2.0.1

# Install basic dependency
pip install -e .

conda install cudatoolkit=11.0
conda install -c nvidia cudnn
# Now you can run the training script of HAIM-DRL in CARLA Environment.
```
After all these steps, launch the CARLA experiment through:
```bash
python train_haim_drl_in_carla.py --num-gpus=1
```
Currently, a steering wheel controller is default to reproduce the CARLA Experiment. 
We also provide keyboard interface for controlling vehicles in CARLA, which can be turned on by setting
```keyboard_control:True``` in the training script.
For providing navigation information, there is a status, namely ```command:```, at the upper-left of the interface.

## Training baselines
### RL baselines 
For SAC/PPO/PPO-Lag/SAC-Lag, there is no additional requirement to run the training scripts. 
```bash
# use previous HAIM-DRL environment
conda activate HAIM-DRL  
cd haim_drl/run_baselines
# launch baseline experiment
python train_[ppo/sac/sac_lag/ppo_lag].py --num-gpus=[your_gpu_num]
```

### Human Demonstration
Human demonstration is required to run Imitation Learning (IL). You can collect human demonstration by runing:
```bash
cd haim_drl/utils
python collect_human_data_set.py
```
or you can use the data collected by our human expert [here](https://github.com/zilin-huang/HAIM-DRL/releases/tag/v0.0.0)

### CQL/BC
If you wish to run CQL, extra setting is required as follows:
```bash
# ray needs to be updated to 1.2.0
pip install ray==1.2.0
cd haim_drl/run_baselines
# launch baseline experiment
python train_cql.py --num-gpus=0 # do not use gpu
```
For BC training, modify the config ```bc_iter=1000000``` in ```train_cql.py``` to convert the CQL into BC, and re-run this script.  

### Human-in-the-loop baselines and GAIL
To run GAIL/HG-DAgger/IWR, please create a new conda environment and install GPU-version of torch:
```bash
# Create virtual environment
conda create -n HAIM-DRL-torch python=3.7
conda activate HAIM-DRL-torch

# Install basic dependency
pip install -e .

# install torch
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
conda install cudatoolkit=11.0
```
Now, IWR/HG-Dagger/GAIL can be trained by:
```bash
cd haim_drl/run_baselines 
python train_[IWR/gail/hg_dagger].py
```



## Reference

```latex
@article{huang2024human,
  title={Human as AI Mentor: Enhanced Human-in-the-loop Reinforcement Learning for Safe and Efficient Autonomous Driving},
  author={Huang, Zilin and Sheng, Zihao and Ma, Chengyuan and Chen, Sikai},
  journal={arXiv preprint arXiv:2401.03160},
  year={2024}
}
```
