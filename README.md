# Soft Actor-Critic (SAC) Reinforcement Learning

## Overview

This project implements the **Soft Actor-Critic (SAC)** algorithm.
The implementation is compatible with environments from **OpenAI Gym** and can be adapted to custom continuous control tasks.

---

## Usage

```
usage: main.py [-h] [--env ENV] [--max_timesteps MAX_TIMESTEPS] [--save_model] [--eval_freq EVAL_FREQ] [--batch_size BATCH_SIZE] [--alpha ALPHA] [--lr LR] [--discount DISCOUNT] [--tau TAU] [--fixed_alpha]

options:
  -h, --help            show this help message and exit
  --env ENV
  --max_timesteps MAX_TIMESTEPS
                        total timesteps of the experiments
  --save_model
  --eval_freq EVAL_FREQ
  --batch_size BATCH_SIZE
  --alpha ALPHA         Determines the relative importance of the entropy term
  --lr LR               the learning rate of the optimizer
  --discount DISCOUNT   Discount factor.
  --tau TAU             Target network update rate
  --fixed_alpha
```
---

![Score](images/1.png?raw=true)
![Score](images/2.png?raw=true)
![Demo](images/1.gif?raw=true)
![Demo](images/2.gif?raw=true)
![Demo](images/6.gif?raw=true)
![Demo](images/3.gif?raw=true)
![Demo](images/4.gif?raw=true)
![Demo](images/5.gif?raw=true)
