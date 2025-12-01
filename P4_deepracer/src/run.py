import yaml
import time
import torch
import torch.nn as nn
import datetime
import numpy as np
from pathlib import Path
from loguru import logger
from munch import munchify
from torch.utils.tensorboard import SummaryWriter
from gymnasium.vector import AsyncVectorEnv

from src.agents import MyFancyAgent, RandomAgent
from src.buffer import RolloutBuffer
from src.utils import (
    device,
    set_seed,
    make_environment,
)


DEVICE = device()
ROOT_DIR = Path(__file__).resolve().parent.parent
HYPER_PARAMS_PATH = ROOT_DIR / 'configs' / 'hyper_params.yaml'
RUNS_DIR = ROOT_DIR / 'runs'
CHECKPOINTS_DIR = ROOT_DIR / 'checkpoints'
MODELS_DIR = ROOT_DIR / 'models'


def tensor(x: np.array, type=torch.float, device=DEVICE) -> torch.Tensor:
    return torch.tensor(x, dtype=type, device=device)


def zeros(x: tuple, type=torch.float, device=DEVICE) -> torch.Tensor:
    return torch.zeros(x, dtype=type, device=device)


def _to_scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 0:
            raise ValueError("Tensor has no elements to convert to scalar.")
        return value.view(-1)[0].item()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("Array has no elements to convert to scalar.")
        return float(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError("Sequence has no elements to convert to scalar.")
        return float(value[0])
    if hasattr(value, 'item'):
        try:
            return float(value.item())
        except Exception:
            pass
    return float(value)


def run(hparams):
    start_time = time.time()
    
    # Load hyper-params
    with HYPER_PARAMS_PATH.open('r') as file:
        default_hparams = yaml.safe_load(file)
    
    final_hparams = default_hparams.copy()
    final_hparams.update(hparams)
    args = munchify(final_hparams)
    
    # Ensure output directories exist
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    run_name = (
        f"{args.environment}__{args.experiment_name}__{args.seed}__{int(time.time())}"
    )
    writer = SummaryWriter(str(RUNS_DIR / run_name))
    writer.add_text(
        'hyperparameters',
        "|param|value|\n|-|-|\n%s" % (
            "\n".join(
                [f"|{key}|{value}|" for key, value in vars(args).items()]
            )
        ),
    )
    
    set_seed(args.seed)
    
    # Setup environment and agent
    host = getattr(args, 'host', '127.0.0.1')
    base_port = getattr(args, 'base_port', 8888)
    
    if args.num_envs == 1:
        env = make_environment(args.environment, seed=args.seed, host=host, port=base_port)
        obs_space = env.observation_space
        action_space = env.action_space
    else:
        def _make_env(index):
            port = base_port + index
            seed_offset = args.seed + index

            def _init():
                return make_environment(
                    args.environment,
                    seed=seed_offset,
                    host=host,
                    port=port
                )
            return _init
        
        env_fns = [_make_env(i) for i in range(args.num_envs)]
        env = AsyncVectorEnv(env_fns)
        obs_space = env.single_observation_space
        action_space = env.single_action_space
    
    obs_dim = np.prod(obs_space.shape)
    
    # Initialize agent
    agent = MyFancyAgent(obs_space, action_space, name=args.experiment_name).to(DEVICE)

    pretrained_model = getattr(args, 'pretrained_model', None)
    optimizer_state = None
    if pretrained_model:
        checkpoint_path = Path(pretrained_model)
        if not checkpoint_path.is_absolute():
            checkpoint_path = ROOT_DIR / checkpoint_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Pretrained model not found at {checkpoint_path}"
            )
        state = torch.load(checkpoint_path, map_location=DEVICE)
        if isinstance(state, dict) and 'agent_state_dict' in state:
            optimizer_state = state.get('optimizer_state_dict')
            state = state['agent_state_dict']
        agent.load_state_dict(state)
        logger.info(f"Loaded pretrained weights from {checkpoint_path}")

    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        logger.info("Loaded optimizer state from checkpoint")
    
    # Initialize rollout buffer
    buffer = RolloutBuffer(
        num_steps=args.num_steps,
        num_envs=args.num_envs,
        obs_dim=obs_dim,
        device=DEVICE
    )
    
    # Training loop
    global_step = 0
    num_updates = args.total_timesteps // (args.num_steps * args.num_envs)
    
    obs, info = env.reset(seed=args.seed)
    obs = tensor(obs).unsqueeze(0) if args.num_envs == 1 else tensor(obs)  # [num_envs, obs_dim]
    
    logger.info(f"Starting PPO training for {num_updates} updates ({args.total_timesteps} total steps)")
    
    for update in range(1, num_updates + 1):
        # ============================================
        # ROLLOUT COLLECTION
        # ============================================
        agent.eval()
        
        for step in range(args.num_steps):
            global_step += args.num_envs
            
            with torch.no_grad():
                actions, log_probs, _, values = agent.get_action_and_value(obs)
            
            # Environment step
            if args.num_envs == 1:
                next_obs, reward, terminated, truncated, info = env.step(actions.item())
                done = terminated or truncated
                
                # Log episode statistics before reset
                if done and isinstance(info, dict) and 'episode' in info:
                    ep_return = _to_scalar(info['episode']['r'])
                    ep_length = _to_scalar(info['episode']['l'])
                    writer.add_scalar('rollout/episodic_return', ep_return, global_step)
                    writer.add_scalar('rollout/episodic_length', ep_length, global_step)
                    logger.info(
                        f"Update {update}/{num_updates}, Step {global_step}: "
                        f"Episode Return = {ep_return:.2f}, Episode Length = {ep_length}"
                    )
                    # Manual reset to avoid RecordEpisodeStatistics assertion
                    next_obs, info = env.reset()
                
                next_obs = tensor(next_obs).unsqueeze(0)
            else:
                next_obs, rewards, terminateds, truncateds, infos = env.step(actions.cpu().numpy())
                dones = np.logical_or(terminateds, truncateds)
                next_obs = tensor(next_obs)
                reward = torch.as_tensor(rewards, dtype=torch.float32, device=DEVICE)
                done = torch.as_tensor(dones, dtype=torch.bool, device=DEVICE)

                info_records = []
                if isinstance(infos, dict):
                    final_infos = infos.get('final_info')
                    if final_infos is not None:
                        info_records.extend([fi for fi in final_infos if fi])
                    episode_info = infos.get('episode')
                    if episode_info is not None:
                        if isinstance(episode_info, dict):
                            info_records.append(episode_info)
                        elif isinstance(episode_info, (list, tuple)):
                            info_records.extend([ep for ep in episode_info if ep])
                elif isinstance(infos, (list, tuple)):
                    info_records.extend(infos)

                for info_item in info_records:
                    ep_info = None
                    if isinstance(info_item, dict) and 'episode' in info_item:
                        ep_info = info_item['episode']
                    elif isinstance(info_item, dict) and all(k in info_item for k in ('r', 'l')):
                        ep_info = info_item

                    if ep_info is not None:
                        ep_return = _to_scalar(ep_info['r'])
                        ep_length = _to_scalar(ep_info['l'])
                        writer.add_scalar('rollout/episodic_return', ep_return, global_step)
                        writer.add_scalar('rollout/episodic_length', ep_length, global_step)
                        logger.info(
                            f"Update {update}/{num_updates}, Step {global_step}: "
                            f"Episode Return = {ep_return:.2f}, Episode Length = {ep_length}"
                        )
            
            # Store transition
            buffer.add(obs, actions, log_probs, reward, done, values)
            
            obs = next_obs
        
        # Compute advantages with GAE
        with torch.no_grad():
            next_value = agent.get_value(obs)
        buffer.compute_gae(next_value, gamma=args.gamma, gae_lambda=args.gae_lambda)
        
        # ============================================
        # PPO UPDATE
        # ============================================
        agent.train()
        
        # Track metrics for logging
        update_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'approx_kl': [],
            'clip_fraction': [],
        }
        
        for epoch in range(args.num_epochs):
            for batch in buffer.get_batches(args.batch_size):
                # Get current policy outputs
                _, new_log_probs, entropy, new_values = agent.get_action_and_value(
                    batch['obs'], batch['actions']
                )
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(new_log_probs - batch['log_probs'])
                clip_ratio = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
                policy_loss = -torch.min(
                    ratio * batch['advantages'],
                    clip_ratio * batch['advantages']
                ).mean()
                
                # Value loss (MSE)
                new_values = new_values.squeeze(-1) if new_values.dim() > 1 else new_values
                value_loss = nn.functional.mse_loss(new_values, batch['returns'])
                
                # Entropy bonus
                entropy_loss = entropy.mean()
                
                # Combined loss
                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy_loss
                
                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                
                # Track metrics
                with torch.no_grad():
                    approx_kl = (batch['log_probs'] - new_log_probs).mean()
                    clip_frac = ((ratio - 1.0).abs() > args.clip_eps).float().mean()
                
                update_metrics['policy_loss'].append(policy_loss.item())
                update_metrics['value_loss'].append(value_loss.item())
                update_metrics['entropy'].append(entropy_loss.item())
                update_metrics['approx_kl'].append(approx_kl.item())
                update_metrics['clip_fraction'].append(clip_frac.item())
        
        # Log averaged metrics
        writer.add_scalar('train/policy_loss', np.mean(update_metrics['policy_loss']), global_step)
        writer.add_scalar('train/value_loss', np.mean(update_metrics['value_loss']), global_step)
        writer.add_scalar('train/entropy', np.mean(update_metrics['entropy']), global_step)
        writer.add_scalar('train/approx_kl', np.mean(update_metrics['approx_kl']), global_step)
        writer.add_scalar('train/clip_fraction', np.mean(update_metrics['clip_fraction']), global_step)
        writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], global_step)
        
        # Compute explained variance
        with torch.no_grad():
            y_pred = buffer.values.flatten()
            y_true = buffer.returns.flatten()
            var_y = torch.var(y_true)
            explained_var = 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)
        writer.add_scalar('train/explained_variance', explained_var.item(), global_step)
        
        # Reset buffer for next rollout
        buffer.reset()
        
        # Periodic logging
        if update % 10 == 0:
            elapsed_time = time.time() - start_time
            elapsed_str = str(datetime.timedelta(seconds=round(elapsed_time)))
            logger.info(
                f"Update {update}/{num_updates} | "
                f"Step {global_step}/{args.total_timesteps} | "
                f"Policy Loss: {np.mean(update_metrics['policy_loss']):.4f} | "
                f"Value Loss: {np.mean(update_metrics['value_loss']):.4f} | "
                f"Approx KL: {np.mean(update_metrics['approx_kl']):.4f} | "
                f"Time: {elapsed_str}"
            )
        
        # Save checkpoint periodically
        if update % args.save_interval == 0:
            checkpoint_path = CHECKPOINTS_DIR / f"{run_name}_update_{update}.pt"
            torch.save({
                'update': update,
                'global_step': global_step,
                'agent_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    # Final save
    final_path = MODELS_DIR / f"{run_name}_final.pt"
    torch.save(agent.state_dict(), final_path)
    logger.info(f"Training complete. Final model saved: {final_path}")
    
    env.close()
    writer.close()
