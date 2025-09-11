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
## Demo
<table width="100%">
  <tr>
    <td width="33%">
      <img src="images/3.png?raw=true" alt="Learn" width="100%">
    </td>
    <td width="33%">
      <img src="images/1.png?raw=true" alt="Score" width="100%">
    </td>
    <td width="33%">
      <img src="images/2.png?raw=true" alt="Score" width="100%">
    </td>
  </tr>
</table>
---
<table width="100%">
  <tr>
    <td width="33%">
      <video src="https://github.com/user-attachments/assets/85caa14a-83f8-47e9-b707-d59fff0da897" width="100%" controls></video>
    </td>
    <td width="33%">
      <video src="https://github.com/user-attachments/assets/61f68d6d-ee64-4026-9fcf-f4ea1a711224" width="100%" controls></video>
    </td>
    <td width="33%">
      <video src="https://github.com/user-attachments/assets/2f9e1a6e-e378-488b-91ed-e87c68370ae2" width="100%" controls></video>
    </td>
  </tr>
  <tr>
    <td width="33%">
      <video src="https://github.com/user-attachments/assets/18e72682-ea92-4fbe-a492-1074723e9436" width="100%" controls></video>
    </td>
    <td width="33%">
      <video src="https://github.com/user-attachments/assets/4b0875d2-6b8c-44b8-99d1-0657f1d5f086" width="100%" controls></video>
    </td>
    <td width="33%">
      <video src="https://github.com/user-attachments/assets/0ab9726b-de3a-4879-8417-f89a6c8e176a" width="100%" controls></video>
    </td>
  </tr>
</table>
