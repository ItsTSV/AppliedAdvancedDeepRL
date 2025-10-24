## Reinforcement learning for robotics and complex simulated environments
This repository contains implementation of my Diploma thesis focused on Deep Reinforcement Learning. So far, it is still
heavily work in progress, but there are some parts already done. The environments I want to solve are mainly from robotics
domain; though there might be some other simulated environments as well, if there is enough time and computational resources ;)

### Implemented algorithms:
- [x] Proximal Policy Optimization (PPO)
- [x] Soft Actor-Critic (SAC)
- [x] Rainbow DQN (might be swapped out for some other algorithm, such as TD3 or DDPG later on)

### Environments to solve:
- [ ] MuJoCo
  -  [x] Swimmer-v5
- [ ] Gymnasium-Robotics

### Trained models:
I already trained some models. Storing them in GitHub repository ain't exactly a good idea; I will temporarily do it, but 
later on, models will be stored on Hugging Face instead. You can find them [here](https://huggingface.co/collections/ItsTSV/reinforcement-learning-in-robotic-and-simulated-environments).

### Training logs and videos:
Training logs and videos will be stored on Weights & Biases, later on Streamlit presentation page. For now, they remain
private.

### Commands
To ensure compactibility of all libraries (mainly torch and its components: torchrl, torchvision etc...), requirements.txt
are used. For formatting and linting, black and pylint are used. Black sometimes loves to produce a bit weird looking code,
but it is a standard in Python ecosystem, so the code looks weird pretty much everywhere ;)

Install all dependencies: ```pip install -r requirements.txt```

Format source code: ```black src```

Run linter: ```pylint src```

### Random gifs
... because looking at images is more interesting than reading text.
