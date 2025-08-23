## Reinforcement learning for robotics and complex  environments

Proximal Policy Optimization
- [x] Implemented
- [x] Code level optimised
- [x] Supports continuous and discrete action spaces
- [x] Tested
- [ ] Used in robotics and complex simulated environments

---

Soft Actor Critic
- [x] Implemented
- [ ] Code level optimised
- [ ] Tested
- [ ] Used in robotics

---

Rainbow DQN
- [ ] Implemented
  - [x] DQN
  - [x] Double DQN
  - [x] Dueling DQN
  - [x] Prioritised Experience Replay
  - [ ] Noisy Nets
  - [ ] Multi-step learning
  - [ ] Distributional RL
- [ ] Code level optimised
- [x] Tested for implemented parts
- [ ] Used in complex simulated environments

--- 

### Commands
To ensure compactibility of all libraries (mainly torch and its components: torchrl, torchvision etc...), requirements.txt
are used. For formatting and linting, black and pylint are used. Black sometimes loves to produce a bit weird looking code,
but it is a standard in Python ecosystem, so the code looks weird pretty much everywhere ;)

Install all dependencies: ```pip install -r requirements.txt```

Format source code: ```black src```

Run linter: ```pylint src```