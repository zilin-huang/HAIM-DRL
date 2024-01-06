# Human as AI mentor-based deep reinforcement learning (HAIM-DRL)

Human as AI Mentor: Enhanced Human-in-the-loop Reinforcement Learning for Safe and Efficient Autonomous Driving

Code Upload in Progress

We're actively organizing our code for this repository. It takes some time to ensure everything is ready, but we're committed to completing the upload by JAN. Stay tuned for updates!



## Installation

```bash
# Clone the code to local
[git clone https://github.com/decisionforce/HAIM-DRL.git](https://github.com/zilin-huang/HAIM-DRL.git)
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
Since the main experiment of HAIM-DRL takes one hour and requires a steering wheel (Logitech G29), we further provide an 
easy task for users to experience HAIM-DRL.
```bash
cd HAIM-DRL/run_main_exp/
python train_HAIM-DRL_keyboard_easy.py --num-gpus=1
```
In this task, human is authorized to take over the vehicle by pressing **W/A/S/D** and guide or safeguard the agent to 
the destination ("E" can be used to pause simulation). 
Since there is only one map in this task, 10 minutes or 5000 transitions is enough for HAIM-DRL agent to learn a policy.

### Main Experiment
To reproduce the main experiment reported in paper, run following scripts:
```bash
python train_HAIM-DRL.py --num-gpus=1
```
If steering wheel is not available, set ```controller="keyboard"``` in the script to train HAIM-DRL agent. After launching this script,
one hour is required for human to assist HAIM-DRL agent to learn a generalizable driving policy by training in 50 different maps.
