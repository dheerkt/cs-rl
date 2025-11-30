import yaml
import time
import torch
import torch.nn as nn
import datetime
import numpy as np
from loguru import logger
from munch import munchify
from torch.utils.tensorboard import SummaryWriter

from src.agents import MyFancyAgent, RandomAgent
from src.buffer import RolloutBuffer
from src.utils import (
    device,
    set_seed,
    make_environment,
)


DEVICE = device()
HYPER_PARAMS_PATH: str='configs/hyper_params.yaml'


def tensor(x: np.array, type=torch.float, device=DEVICE) -> torch.Tensor:
    return torch.tensor(x, dtype=type, device=device)


def zeros(x: tuple, type=torch.float, device=DEVICE) -> torch.Tensor:
    return torch.zeros(x, dtype=type, device=device)


def run(hparams):
    start_time = time.time()
    
    # Load hyper-params
    with open(HYPER_PARAMS_PATH, 'r') as file:
        default_hparams = yaml.safe_load(file)
    
    final_hparams = default_hparams.copy()
    final_hparams.update(hparams)
    args = munchify(final_hparams)
    
    # Setup logging
    run_name = (
        f"{args.environment}__{args.experiment_name}__{args.seed}__{int(time.time())}"
    )
    writer = SummaryWriter(f"runs/{run_name}")
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
    env = make_environment(args.environment)
    
    # Get observation and action space dimensions
    obs_space = env.observation_space
    action_space = env.action_space
    obs_dim = np.prod(obs_space.shape)
    
    # Initialize agent
    agent = MyFancyAgent(obs_space, action_space, name=args.experiment_name).to(DEVICE)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
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
    
    obs, info = env.reset()
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
                    writer.add_scalar('rollout/episodic_return', info['episode']['r'], global_step)
                    writer.add_scalar('rollout/episodic_length', info['episode']['l'], global_step)
                    logger.info(
                        f"Update {update}/{num_updates}, Step {global_step}: "
                        f"Episode Return = {info['episode']['r']:.2f}, "
                        f"Episode Length = {info['episode']['l']}"
                    )
                    # Manual reset to avoid RecordEpisodeStatistics assertion
                    next_obs, info = env.reset()
                
                next_obs = tensor(next_obs)
            else:
                next_obs, rewards, terminateds, truncateds, infos = env.step(actions.cpu().numpy())
                dones = np.logical_or(terminateds, truncateds)
                next_obs = tensor(next_obs)
                reward = rewards
                done = dones
            
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
            checkpoint_path = f"checkpoints/{run_name}_update_{update}.pt"
            torch.save({
                'update': update,
                'global_step': global_step,
                'agent_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    # Final save
    final_path = f"models/{run_name}_final.pt"
    torch.save(agent.state_dict(), final_path)
    logger.info(f"Training complete. Final model saved: {final_path}")
    
    env.close()
    writer.close()
