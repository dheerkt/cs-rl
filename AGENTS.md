# DeepRacer: Optimization and Adaptation Project

## Project Context & LLM Instructions

### Current Implementation Status
- **Phase:** Part I - Time Trial Agent Development
- **Algorithm:** PPO with custom CNN encoder
- **Architecture:** 3-layer CNN + MLP heads for Actor/Critic
- **Training Strategy:** Vectorized training with AsyncVectorEnv (12 workers)
- **Current Track:** reInvent2019 (alternating directions)

### Active Task List
1. [ ] Implement PPO algorithm with GAE
2. [ ] Build CNN encoder with orthogonal initialization
3. [ ] Create vectorized training pipeline
4. [ ] Design Time Trial reward function
5. [ ] Train baseline agent on reInvent2019
6. [ ] Evaluate across all 3 tracks for robustness

### LLM Instructions
- Use this document as technical reference for implementation
- Follow Section 6 Implementation Directives for hyperparameters
- Avoid Section 5 pitfalls during development
- Prioritize completion rate over lap time initially
- Document experiments and design decisions

---

## Contents
1 Overview
2 AWS DeepRacer
2.1 State and Observation Spaces
2.2 Action Space
2.3 Reward Function
2.4 Termination Conditions
2.5 Environment and Tasks
3 The Problem
3.1 Part I: Robust Time-Trial Across Tracks
3.2 Part II: Adaptation to Other Race Types
4 Implementation Details
5 Technical Constraints & Common Pitfalls
6 Implementation Directives (LLM Constraints)
7 Resources
7.1 Documentation

## 1 Overview
In this project, you will explore the challenges of training agents for an autonomous racing car using a local
variant of AWS DeepRacer (Balaji et al. 2019; Balaji et al. 2020). It will introduce you to practical applications
of reinforcement learning in robotics while highlighting the complexities of training agents capable of generalizing
to different environments and tasks. Your main objective is to develop and refine models capable of solving
three distinct race types: Time Trial, Object Avoidance, and Head-to-Head. Each race type presents unique
challenges, from optimizing speed on an open track to navigating around obstacles and competing against other
agents. You will use reward function design, representation learning, hyperparameter tuning and incremental
training methods to improve model performance across varied tracks and scenarios.

## 2 AWS DeepRacer
The AWS DeepRacer vehicle is a 1 /18th scale autonomous racing car equipped with front-facing cameras for
stereo vision, and LIDAR sensors, which enhance obstacle detection and depth perception. This sensor suite,
combined with customizable reward functions, enables you to develop and test drive models that navigate real-
world track scenarios using a deep reinforcement learning model learned from virtual training. For ease of use, in
this project we provide a local simulation of the AWS DeepRacer environment together with a deepracer gym
package for a familiar gymnasium API via the deepracer-v0 environment.

### 2.1 State and Observation Spaces
At each time-step, a typical vehicle's state on a track *could* be represented as a tuple of state variables such as
(x, y, θ, v, ˙θ, p, s progress ),
where
•xandyrepresent the position of the DeepRacer vehicle on the track.
•θis the heading angle of the vehicle, indicating its orientation relative to the track.
•vis the current speed of the vehicle.
•˙θis the angular velocity, indicating the rate of change in the vehicle's heading.
•pis the vehicle's distance from the centerline of the track.
•sprogress is the cumulative progress percentage along the track.

However, these (or other state variables) may be difficult to measure or entirely inaccessible for a real vehicle
on an arbitrary track and are therefore not included in the observation space. Observations are only made
via measurements from the sensors that a car is equipped with , which include
•Camera sensors: You can choose one of two front-facing camera sensor settings – a color monocular
camera or greyscale stereo cameras as shown in Fig. 1. At each time-step, we receive a measurement from
the selected camera(s) as an image(s).
•LIDAR sensor: You can choose to include an optional LIDAR sensor that outputs a 64 dimensional
vector representing ranges, i.e. radial distances (meters), to the surrounding obstacles. The 64 readings
are uniformly spread over 360◦, starting from ∼ −2.8◦(to∼+2.8◦) counterclockwise. Note that LIDAR
measurements beyond a minimum of 0 .15m and a maximum of 1m are undefined.

Please consult the AWS DeepRacer developer guide for more details about different sensors and their possible
use cases. For the purposes of this project, we shall restrict ourselves to the stereo cameras and LIDAR for all
of the following experiments. These can be specified in the configs/agent params.json file. Please see the
deepracer gympackage documentation for instructions and examples on specifying sensors.

Generally, these sensor measurements are not sufficient to recover the state of the vehicle, and therefore
the MDP is only partially observable. Nevertheless, some state variables together with additional auxiliary
variables are accessible within the reward function to calculate the rewards. The same is also available in the
auxiliary information returned by the gymnasium.Env.step function for debugging and evaluation purposes.
For a complete set of these state and auxiliary variables, please see the AWS DeepRacer developer guide.

### 2.2 Action Space
The agent can drive the DeepRacer vehicle by adjusting the throttle to control the speed and adjusting the
steering angle to control the direction. This can be defined in the configs/agent params.json file via a
discrete or continuous action space as follows.
•Discrete: The action space is a set of integers a∈ {1,···, n}that enumerate a set of n∈Z+pre-defined
tuples/combinations of speed and steering angle.
•Continuous: The action space is a 2D vector representing speed and steering angle within set ranges.

See the deepracer gympackage documentation for instructions and examples on setting the action space.

### 2.3 Reward Function
The reward function must be defined in configs/reward function.py in accordance with the task at hand.
You can find an exhaustive list of the reward function parameters as well as reward function examples in the
AWS DeepRacer developer guide. Note that these examples by themselves may not suffice for achieving the
best possible performance (e.g., lap-time), but may serve as an inspiration for you to define your own reward
function or make any modifications thereof.

### 2.4 Termination Conditions
The provided DeepRacer simulation terminates each episode when any of the following conditions are satisfied.
•Lap completed: When the agent finishes one lap around the track.
•Crashed: When the agent collides with an obstacle or a bot car.
•Off track: When the agent goes completely off of the track (i.e. none of the wheels are on the track).
•Reversed: When the agent goes in the opposite direction for 15 consecutive steps.

The first three are accessible in the reward function1. Furthermore, the simulation truncates the episode on the
following conditions.
•Immobilized: When the agent moves less than 0 .3 mm for 15 consecutive steps.
•Time up: When the agent reaches a maximum number of 100k steps in an episode.

### 2.5 Environment and Tasks
The environment, including the race track, race type and other relevant parameters can be defined in the
configs/environment params.yaml file. We shall discuss some of these configuration parameters below, but
for details please see the DeepRacer-for-Cloud reference manual.

#### 2.5.1 Race Types
You can simulate the following types of racing events by appropriately configuring the environment.
•Time-Trial: Race against the clock on an unobstructed track and aim to get the fastest lap time possible.
SetNUMBER OFOBSTACLES to 0 and NUMBER OFBOTCARS to 0.
•Obstacle-Avoidance: Race against the clock on a track with stationary obstacles and aim to get the
fastest lap time possible. Set NUMBER OFOBSTACLES to 6 and NUMBER OFBOTCARS to 0.
•Head-to-Bot: Race against one or more other vehicles on the same track and aim to cross the finish line
while avoiding other vehicles. Set NUMBER OFOBSTACLES to 0 and NUMBER OFBOTCARS to 3.

#### 2.5.2 Evaluation Metrics
In this project we shall restrict ourselves to the race tracks in Fig. 2 by setting the WORLD NAME variable accord-
ingly. You are only allowed to use these for training and evaluation purposes. For any race type, the problem
is considered solved when the agent can traverse 100% of the track (i.e., completes a lap) for 5 consecutive
episodes across all 3 tracks. This is to ensure that the policy is robust across tracks and not just overfitted to
the training track. We have provided a src.utils.evaluate function to make such an evaluation easier.
Additionally, we also provide calculation of lap-times for each evaluation run in src.utils.evaluate .
These can be used as a tie-breaker to pick the best agent if multiple agents are able to solve the problem.

## 3 The Problem
This project is divided into two sequential parts, with each part building upon the previous one. In Part I,
you will focus on developing an agent to solve the Time-Trial problem, optimizing its performance on specific
tracks through custom reward functions and hyperparameter tuning. The best-performing models from Part I
will then be used as the foundation for Part II, where you will apply further refinements and testing to achieve
robust, competitive results across different race types.

### 3.1 Part I: Robust Time-Trial Across Tracks
In this part, your goal is to train an agent that can achieve the fastest possible lap times on the three specified
tracks without any obstacles or competitor bots.

You will first design your solution to have an agent at least solve the simplest of the three project tracks.
Once you have a working solution, you can decide to switch to the more challenging tracks if required.
Furthermore, since we can only pick one track at a time during training, you will need to find out what it
takes to have a single agent solve all of the tracks.

#### 3.1.1 Solution Design
You will define and train your agent using the provided deepracer gymenvironment and associated tools. This
involves specifying several key components:
•Action Space: Define an action space as either discrete or continuous with appropriate parameterization.
•Reward Function: Design a custom reward function to solve the Time-Trial task. This is crucial for
guiding the agent's learning.
•RL Algorithm: You may use any algorithm that you consider suitable for this problem.
•Function Space: Define the function space for the policy/value function approximation. This also
includes any pre-processing you may perform on the observation samples in addition to the architecture
of the neural-network, if used.
•Hyper-Parameters: Enumerate the key hyper-parameters for your implemented solution. Also specify
the race track that you used for training your agent. Outline your training schedule if you choose to train
the agent on multiple tracks separately or if you iteratively retrained your agent on all tracks. Try to
choose a set of hyper-parameters that work independently of the choice of training track and/or schedule.

#### 3.1.2 Results and Analysis
Training. While training your Time-Trial agent(s):
•Choose any 3 metrics that you think are helpful to track during training. Plot them against the (global)
number of steps executed by your algorithm.
•Analyze their trends and discuss whether the behavior aligns with your reward function's intent (e.g., is
the agent's lap-time increasing or decreasing?).
•Discuss any challenges encountered during training (e.g., convergence issues, etc.) and how you ad-
dressed them.

Evaluation. After training and tuning your Time Trial agent(s)
•Use the provided src.utils.evaluate function for evaluation and write a commentary on the results.
•Visualize and discuss the agent's behavior using the provided src.utils.demo function. Make sure to
save the video file in your repository.

### 3.2 Part II: Adaptation to Other Race Types
Building on your solution from Part I, this part explores adapting it for the more complex tasks of Object-
Avoidance and Head-to-Bot race types. You may try any approach you think is appropriate, such as changing
the components of your solution from Part I, using your Time-Trial agent to initialize your Object-Avoidance/
Head-to-Bot agents to avoid complete re-training, or a combination thereof.

The main motivation is for you to learn about and experience the non-trivial and open-ended nature of most
real-world RL tasks.

#### 3.2.1 Adaptation, Additional Training and Adjustments
For each of the Object-Avoidance and Head-to-Bot race types, document and justify any changes you made to
your solution from Section 3.1.1.

#### 3.2.2 Results and Analysis
Repeat the training and evaluation analyses of Section 3.1.2 for each of the Object Avoidance and Head-to-
Bot agents. This includes:
•Training. Select and plot 3 relevant metrics for each agent. Analyze their trends and highlight any
training challenges encountered.
•Evaluation. Assess each agent's performance using the src.utils.evaluate function and provide a
brief commentary. Record and save a demo video using src.utils.demo . Additionally, evaluate your
trained Time-Trial agent from Part I on these new tasks to use as a performance baseline.

## 4 Implementation Details
Clone the project repository from https://github.gatech.edu/rldm/P4_deepracer and follow the detailed
instructions in SETUP.md to prepare your development environment. In case you are unsuccessful in setting up
your environment for whatever reason, please follow the instructions to use a PACE ICE machine instead.

Please make sure that your implementation complies with the following development guidelines.

### Provided Code: Familiarize yourself with the provided codebase.
–configs/ : Contains configuration files for the agent, environment, default hyperparameters, and
the reward function. You will modify these extensively. Please make sure to save these for working
solutions either by making a copy or logging via tensorboard .
–src/ : Contains the implementation for your solution in addition to the agent definition ( agents.py ),
function approximation ( transforms.py ), and utility functions ( utils.py ).
–scripts/ : Includes scripts to start and stop the DeepRacer simulator.
–packages/deepracer gym/ : Core wrapper exposing gymnasium API for the DeepRacer simulation.

### Starting Point: A dummy notebook usage.ipynb is provided to demonstrate basic interaction with the environment, policy visualization and evaluation.

### Environment: Keep the following in mind when using the deepracer-v0 Gymnasium environment.
–Vectorization Strategy: While the base environment is not vectorizable by default, use `gymnasium.vector.AsyncVectorEnv` to run multiple parallel instances (recommended: 12 workers) to accelerate data collection.
–Observation Handling: The deepracer-v0 environment returns observations as a dictionary which is
less straightforward to handle as opposed to just vectors. Use the provided src.utils.make environment
function instead which 'flattens' the observations into a vector. If you wish to 'un-flatten' the obser-
vations at any point, you may use the provided src.transforms.UnflattenObservation class.

### Agent: Make sure to inherit all of your agents from the src.agents.Agent class structure provided in
src/agents.py to ensure compatibility with the utility functions.

### Function Approximation: You may use any function approximator to solve this problem, however we
strongly recommend you use a non-linear function approximator such as a convolutional neural network
(CNN). If you decide to train a neural network, you must train it using PyTorch as an automatic
differentiation tool. Other neural network training libraries such as Tensorflow, Keras, Theano are not
allowed. PyTorch is well liked amongst researchers for its pythonic feel. We have provided a default CNN
encoder under src.transforms as an example. You may make any changes that you deem appropriate.

### Workflow: Use separate notebooks or scripts for Part I and Part II to keep your experiments organized.
Use tensorboard to monitor training progress and save configurations/hyper-parameters.

## 5 Technical Constraints & Common Pitfalls

### Critical Infrastructure Issues
*   **Evaluation Strategy:** Do NOT use `src.utils.evaluate` during training loops. It causes ZMQ/Docker crashes by trying to restart containers dynamically. Use a custom inline evaluation loop on the active track instead.
*   **Container Hygiene:** Always run cleanup before starting new training sessions to kill orphan containers that cause "Address already in use" errors.
    *   *Command:* `docker rm -f $(docker ps -aq)` or `kill -9 $(lsof -ti :[GYM_PORT])`
*   **Graph Handling:** If sharing a CNN backbone between Actor/Critic, use a combined loss function (`loss = policy_loss + value_loss`) to avoid `RuntimeError: Trying to backward through the graph a second time`.

### Environment-Specific Issues
*   **Directionality:** The environment alternates driving direction by default. Ensure your agent is trained on both directions (or use symmetric data augmentation) to prevent "Left Turn Only" policies that fail 50% of the time.
*   **Speed vs. Completion:** Prioritize stability over speed. The primary goal is 100% completion rate. Only optimize for lap times after achieving consistent completion.

### Architecture Recommendations
*   **Simpler Vision:** Do not use pre-trained models (ResNet/DINO). They add inference latency that slows data collection. A custom 3-layer CNN is sufficient and faster to train.
*   **Bottleneck Analysis:** The simulator physics, not the vision backbone, is the primary bottleneck. Focus on reward function design and hyperparameter tuning over complex architectures.

## 7 Resources

### 7.1 Documentation
•AWS DeepRacer developer guide: Details on reward function, action spaces, and general concepts.
–Main Page: https://docs.aws.amazon.com/deepracer/latest/developerguide/index.html
–Reward Function Reference: https://docs.aws.amazon.com/deepracer/latest/developerguide/deepracer-reward-function-reference.html
•DRfC: https://aws-deepracer-community.github.io/deepracer-for-cloud/reference.html (for understanding environment parameters).
•deepracer gymPackage: packages/README.md for usage of the local sim. environment wrapper.
•Gymnasium: https://gymnasium.farama.org/ (For understanding the RL environment API).
•PyTorch: https://pytorch.org/docs/stable/index.html
•TensorBoard: https://www.tensorflow.org/tensorboard/ (For visualizing training logs).