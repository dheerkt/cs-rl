# Multi-Agent Overcooked RL Project

## Overview
This project implements a multi-agent reinforcement learning solution for the Overcooked environment using PPO with Centralized Training and Decentralized Execution (CTDE).

## Algorithm
- **Policy**: Independent PPO actors per agent (decentralized execution)
- **Value Function**: Centralized critic observing both agents (centralized training)
- **Reward Shaping**: Fixed shaped rewards encouraging coordination
- **Key Feature**: Agent-index swap correction for proper credit assignment

## Installation

### 1. Install PyTorch
Visit https://pytorch.org and install PyTorch for your system. For example:
```bash
# CPU version (recommended for this project)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# OR GPU version (if you have CUDA)
pip install torch torchvision
```

### 2. Install other dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify installation
```bash
python -c "import overcooked_ai_py; print('Overcooked installed successfully')"
```

## Project Structure
```
cs-rl/
├── README.md
├── requirements.txt
├── src/
│   ├── models.py          # PPO actor and centralized critic networks
│   ├── ppo.py             # PPO algorithm implementation
│   ├── reward_shaping.py  # Reward shaping logic with agent-swap fix
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation script
│   └── utils.py           # Helper functions and logging
├── configs/
│   └── hyperparameters.py # Hyperparameters (same across all layouts)
├── results/
│   ├── models/            # Saved checkpoints
│   ├── logs/              # TensorBoard logs
│   └── graphs/            # Generated plots for report
└── notebooks/
    └── analysis.ipynb     # Jupyter notebook for analysis
```

## Usage

### Training
Train on a specific layout:
```bash
python src/train.py --layout cramped_room --episodes 50000
python src/train.py --layout coordination_ring --episodes 100000
python src/train.py --layout counter_circuit_o_1order --episodes 150000
```

### Evaluation
Evaluate trained agents:
```bash
python src/evaluate.py --layout cramped_room --checkpoint results/models/cramped_room_final.pt --num_episodes 100
```

### Generate Graphs
```bash
python src/evaluate.py --generate_graphs
```

## Hyperparameters (Fixed Across All Layouts)
- Learning rate: 3e-4
- Gamma: 0.99
- GAE lambda: 0.95
- PPO clip: 0.2
- Entropy coefficient: 0.01
- Value loss coefficient: 0.5
- Batch size: 2048
- Mini-batch size: 256
- Epochs per update: 10

## Target Performance
≥7 soups delivered per episode on all three layouts:
- cramped_room
- coordination_ring
- counter_circuit_o_1order

## Key Implementation Details

### Centralized Critic
The value network receives concatenated observations from both agents:
```python
critic_input = torch.cat([obs_agent0, obs_agent1], dim=-1)  # Shape: [batch, 192]
value = critic(critic_input)
```

### Agent-Index Swap Fix
On each episode reset, agents are randomly assigned to starting positions. We swap shaped rewards accordingly:
```python
if env.state.players[0].position != initial_positions[0]:
    shaped_rewards = shaped_rewards[::-1]  # Swap agent 0 and 1 rewards
```

### Reward Shaping
Fixed shaping function used across all layouts:
- +0.5: Onion placed in pot
- +1.0: Cooking started with 3 onions
- +1.5: Soup picked up from pot
- +2.0: Correct delivery
- -0.5: Soup dropped or incorrect interaction
- +20.0: Successful delivery (sparse base reward)

## Metrics Tracked
### Required
- Soups delivered per episode (training and evaluation)

### Collaboration Metrics
1. Pot handoffs: How often agents coordinate on the same pot
2. Idle time: Time each agent spends not moving/interacting

## Report Graphs
1. Training curves: Soups per episode for all 3 layouts
2. Evaluation performance: 100-episode evaluation per layout
3. Pot handoffs over training (coordination metric)
4. Agent idle time over training (efficiency metric)

## Git Workflow
```bash
git add .
git commit -m "Descriptive message"
git push origin main
```

Remember to include the final git commit hash in your report.

## Citation
This project uses the Overcooked-AI environment:
```
@inproceedings{carroll2019utility,
  title={On the Utility of Learning about Humans for Human-AI Coordination},
  author={Carroll, Micah and Shah, Rohin and Ho, Mark K and Griffiths, Tom and Seshia, Sanjit and Abbeel, Pieter and Dragan, Anca},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```
