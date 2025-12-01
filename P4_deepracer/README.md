# Project 4, Spring '25 - DeepRacer

![deepracer](https://github.gatech.edu/rldm/P4_deepracer/assets/78388/86684160-fe6f-4a03-972c-078cd9a9afde)

## Clone this repository
```bash
git clone https://github.gatech.edu/rldm/P4_deepracer.git
cd P4_deepracer
```

## Setup and Install Dependencies
This project requires the following to work.
- Docker or Apptainer.
- Conda (or Python 3.10 or higher).
- Linux or Windows machine with an **Intel CPU**.

Please see the detailed setup instructions in [`SETUP.md`](https://github.gatech.edu/rldm/P4_deepracer/blob/main/SETUP.md).

## Usage

Launch the DeepRacer simulation.
```bash
source scripts/start_deepracer.sh \
    [-C=MAX_CPU; default="3"] \
    [-M=MAX_MEMORY; default="6g"]

# example:
# source scripts/start_deepracer.sh -C "3" -M "6g"
```

Interact with the environment via `gymnasium`.
```python
import gymnasium as gym
import deepracer_gym

env = gym.make('deepracer-v0')

observation, info = env.reset()

observation, reward, terminated, truncated, info = env.step(
    env.action_space.sample()
)
```
See the [packages directory](https://github.gatech.edu/rldm/P4_deepracer/tree/main/packages) and the [`usage.ipynb`](https://github.gatech.edu/rldm/P4_deepracer/tree/main/usage.ipynb) notebook for details.

## Demo Videos

The following walkthrough videos were captured with the final Vegas-trained policy via `src.utils.demo`:

- [Time-Trial (reInvent2019_wide)](demos/demo_timetrial.mp4)
- [Obstacle Avoidance (reInvent2019_wide, 6 obstacles)](demos/demo_obstacle.mp4)
- [Head-to-Bot (reInvent2019_wide, 3 bot cars)](demos/demo_headtobot.mp4)

## Quick CLI Reference

### Simulator Management
- Start single simulator (default track): `sudo bash scripts/start_deepracer.sh`
- Start evaluator on specific track: `sudo bash scripts/start_deepracer.sh -E true -W <WORLD_NAME>`
- Launch 16 parallel containers (c7i.12xlarge): `sudo ./scripts/launch_parallel_c7.sh 16`
- Stop all containers: `sudo docker rm -f $(sudo docker ps -aq --filter name=deepracer_)`

### Training
```bash
cd /home/ec2-user/cs-rl/P4_deepracer
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deepracer
nohup python - <<'PY' > train.log 2>&1 &
from src.run import run
run({})  # override keys here if needed
PY
echo $! > train.pid
```
- Current fine-tune: `WORLD_NAME=reInvent2019_track`, `total_timesteps=500000`, `learning_rate=5e-5`, loading the latest checkpoint.
- Default hyperparameters live in `configs/hyper_params.yaml` (num_envs=16, num_steps=128).
- Logs/PIDs reside in repo root (`train.log`, `train.pid`).
- Stop run: `kill $(cat train.pid)`
- Always clean simulators before relaunching: `sudo docker rm -f $(sudo docker ps -aq --filter name=deepracer_)`

### Evaluation (manual track control)
1. Start target track: `sudo bash scripts/start_deepracer.sh -E true -W reInvent2019_wide`
2. Run evaluation without restarts:
   ```bash
   python - <<'PY'
   import torch
   from src.utils import make_environment, evaluate_track
   from src.agents import MyFancyAgent
   env = make_environment('deepracer-v0')
   obs_space, action_space = env.observation_space, env.action_space
   env.close()
   agent = MyFancyAgent(obs_space, action_space, name='time_trial_ppo')
   agent.load_state_dict(torch.load('models/<model>.pt', map_location='cpu'))
   metrics = evaluate_track(agent, 'reInvent2019_wide', manage_simulator=False, episodes=5)
   print(metrics)
   PY
   ```
3. For full three-track sweep (auto restarts): `python - <<'PY' ... evaluate(agent) ...`
   - Use `evaluate_track(..., manage_simulator=False)` when a simulator is already running; the full `evaluate()` helper restarts Docker per track.

### Utilities
- Minimal smoke test: `python test_pipeline.py`
- TensorBoard: `tensorboard --logdir runs`
- Clean gym ports if stuck: `sudo lsof -ti :8888 | xargs sudo kill -9`