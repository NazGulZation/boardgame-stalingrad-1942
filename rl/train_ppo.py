"""CleanRL Maskable PPO Training Script for Stalingrad 1942 (1 vs 1).

Features:
- Invalid action masking via CategoricalMasked distribution
- Multi-channel spatial ResNet Actor-Critic architecture
- 1v1 Self-play training
- Periodic evaluation against the heuristic AI in ai.py
- Checkpoint saving and TensorBoard metric logging
"""

import argparse
import json
import os
import re
import sys
import random
import time
from distutils.util import strtobool

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs): pass
        def add_text(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def close(self): pass

import ai
from rl.models import StalingradResNet, CategoricalMasked
from rl.stalingrad_env import Stalingrad1v1Env, ACTION_SPACE_SIZE
from rl.agent import RLAgent, evaluate_matchup


def write_status(path, data):
    """Atomically write training status dictionary to JSON file."""
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        pass



def parse_args():
    parser = argparse.ArgumentParser(description="CleanRL Maskable PPO for Stalingrad 1942")
    parser.add_argument("--exp-name", type=str, default="stalingrad_1v1_ppo",
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")

    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=50000,
                        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=4,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the generalized advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--eval-interval", type=int, default=5000,
                        help="timesteps between evaluation matches against heuristic AI")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                        help="directory to save model checkpoints")
    parser.add_argument("--status-file", type=str, default="checkpoints/train_status.json",
                        help="path to write live JSON status updates")
    parser.add_argument("--resume-checkpoint", type=str, default="",
                        help="path to existing checkpoint (.pt) to resume training from")
    parser.add_argument("--train-side", type=str, default="axis", choices=["axis", "soviet"],
                        help="the side to train from its perspective ('axis' or 'soviet')")
    parser.add_argument("--opponent", type=str, default="heuristic", choices=["heuristic", "checkpoint", "self", "pool"],
                        help="the opponent policy: 'heuristic', 'checkpoint', 'self', or 'pool'")
    parser.add_argument("--opponent-checkpoint", type=str, default="",
                        help="path to opponent model checkpoint if --opponent=checkpoint")
    parser.add_argument("--opponents", type=str, default="",
                        help="comma-separated list of opponents in the pool (e.g. 'heuristic,axis_v1.pt')")
    parser.add_argument("--model-name", type=str, default="",
                        help="custom output filename for the saved model checkpoint (e.g. 'my_model.pt')")
    parser.add_argument("--min-lr", type=float, default=5e-5,
                        help="minimum learning rate floor during annealing")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args


def train():
    args = parse_args()
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    os.makedirs(args.save_dir, exist_ok=True)

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # Opponent setup
    opponent_pool = []
    if args.opponents and args.opponents.strip():
        items = [x.strip() for x in args.opponents.split(",") if x.strip()]
        for item in items:
            if item.lower() == "heuristic":
                opponent_pool.append(("Heuristic AI", ai.play_turn))
            else:
                cp_path = item if os.path.isabs(item) else os.path.join(args.save_dir, item)
                if os.path.isfile(cp_path):
                    agent_obj = RLAgent(cp_path, device=device)
                    opponent_pool.append((os.path.basename(item), agent_obj.play_turn))
                else:
                    print(f"Warning: Opponent checkpoint '{cp_path}' not found, skipping.")

    # Fallback to single opponent if no pool specified
    if not opponent_pool:
        if args.opponent == "heuristic":
            opponent_pool.append(("Heuristic AI", ai.play_turn))
        elif args.opponent == "checkpoint" and args.opponent_checkpoint:
            cp_path = args.opponent_checkpoint if os.path.isabs(args.opponent_checkpoint) else os.path.join(args.save_dir, args.opponent_checkpoint)
            if os.path.isfile(cp_path):
                opp_agent = RLAgent(cp_path, device=device)
                opponent_pool.append((os.path.basename(cp_path), opp_agent.play_turn))
            else:
                opponent_pool.append(("Heuristic AI", ai.play_turn))
        elif args.opponent == "self":
            opponent_pool = None
        else:
            opponent_pool.append(("Heuristic AI", ai.play_turn))

    if opponent_pool:
        opp_names = [name for name, _ in opponent_pool]
        print(f"Active Opponent Pool ({len(opponent_pool)}): {', '.join(opp_names)}")
        eval_opp_name = f"Pool ({len(opponent_pool)})" if len(opponent_pool) > 1 else opponent_pool[0][0]
    else:
        print("Active Opponent: Self-play")
        eval_opp_name = "Self-play"

    # Environment setup with round-robin pool offset
    if opponent_pool:
        envs = [
            Stalingrad1v1Env(
                rng=random.Random(args.seed + i),
                train_side=args.train_side,
                opponent_pool=opponent_pool,
                initial_opp_idx=i % len(opponent_pool),
                strict_turn_completion=True,
            ) for i in range(args.num_envs)
        ]
    else:
        envs = [
            Stalingrad1v1Env(
                rng=random.Random(args.seed + i),
                train_side=args.train_side,
                opponent_policy=None,
                strict_turn_completion=True,
            ) for i in range(args.num_envs)
        ]

    # Agent setup
    agent = StalingradResNet().to(device)
    prior_steps = 0
    if args.resume_checkpoint and os.path.isfile(args.resume_checkpoint):
        print(f"Resuming training from checkpoint: {args.resume_checkpoint}")
        raw = torch.load(args.resume_checkpoint, map_location=device, weights_only=True)
        if isinstance(raw, dict) and "model_state_dict" in raw:
            state_dict = raw["model_state_dict"]
            prior_steps = raw.get("total_steps", 0)
        else:
            state_dict = raw
            meta_path = os.path.join(args.save_dir, "checkpoints_meta.json")
            cp_base = os.path.basename(args.resume_checkpoint)
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path, "r") as mf:
                        meta = json.load(mf)
                        if cp_base in meta:
                            prior_steps = meta[cp_base].get("total_steps", 0)
                except Exception:
                    pass
        agent.load_state_dict(state_dict)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage setup
    obs_shape = (15, 10, 12)
    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    masks = torch.zeros((args.num_steps, args.num_envs, ACTION_SPACE_SIZE), dtype=torch.bool).to(device)

    # Initialize environment state
    global_step = 0
    start_time = time.time()
    next_obs_list = []
    next_done_list = []
    next_mask_list = []

    for env in envs:
        o, info = env.reset()
        next_obs_list.append(o)
        next_done_list.append(False)
        next_mask_list.append(info["action_mask"])

    next_obs = torch.tensor(np.array(next_obs_list), dtype=torch.float32).to(device)
    next_done = torch.tensor(np.array(next_done_list), dtype=torch.float32).to(device)
    next_mask = torch.tensor(np.array(next_mask_list), dtype=torch.bool).to(device)

    num_updates = max(1, args.total_timesteps // args.batch_size)
    last_eval_step = 0
    last_win_rate = 0.0
    reward_history = []
    ema_reward = None

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = max(args.min_lr, frac * args.learning_rate)
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            masks[step] = next_mask

            # Action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, masks=next_mask)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Step environments
            step_rewards = []
            new_obs_list = []
            new_done_list = []
            new_mask_list = []

            for env_idx, env in enumerate(envs):
                act_idx = int(action[env_idx].item())
                n_obs, reward, terminated, truncated, info = env.step(act_idx)
                done = terminated or truncated

                step_rewards.append(reward)
                if done:
                    n_obs, reset_info = env.reset()
                    new_mask_list.append(reset_info["action_mask"])
                else:
                    new_mask_list.append(info["action_mask"])

                new_obs_list.append(n_obs)
                new_done_list.append(done)

            rewards[step] = torch.tensor(step_rewards, dtype=torch.float32).to(device)
            next_obs = torch.tensor(np.array(new_obs_list), dtype=torch.float32).to(device)
            next_done = torch.tensor(np.array(new_done_list), dtype=torch.float32).to(device)
            next_mask = torch.tensor(np.array(new_mask_list), dtype=torch.bool).to(device)

        # Bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten the batch
        b_obs = obs.reshape((-1,) + obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_masks = masks.reshape((-1, ACTION_SPACE_SIZE))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds], masks=b_masks[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # Calculate approx_kl http://joschu.net/blog/kl-approx.html
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # Calculate rollout reward and EMA
        rollout_mean_reward = float(rewards.mean().item())
        ema_reward = rollout_mean_reward if ema_reward is None else 0.85 * ema_reward + 0.15 * rollout_mean_reward
        reward_history.append([global_step, round(ema_reward, 4)])
        if len(reward_history) > 2000:
            reward_history = reward_history[-2000:]

        # Record metrics
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/mean_reward", rollout_mean_reward, global_step)
        writer.add_scalar("charts/ema_reward", ema_reward, global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if update % 5 == 0 or update == num_updates:
            print(f"Update {update}/{num_updates} | Step {global_step} | SPS: {int(global_step / (time.time() - start_time))} | Reward: {ema_reward:.4f} | Policy Loss: {pg_loss.item():.4f} | Value Loss: {v_loss.item():.4f}")

        # Periodic evaluation against opponent pool
        if global_step - last_eval_step >= args.eval_interval or update == num_updates:
            last_eval_step = global_step
            eval_agent = RLAgent(agent, device=device)
            pool_eval = {}
            pool_win_rates = []

            targets = opponent_pool if opponent_pool else [("Heuristic AI", ai.play_turn)]
            games_per_opp = 5 if len(targets) > 1 else 10
            side_label = "Axis" if args.train_side == "axis" else "Soviet"

            for opp_name, opp_fn in targets:
                if args.train_side == "axis":
                    eval_res = evaluate_matchup(eval_agent, opp_fn, num_games=games_per_opp)
                    wr = eval_res["axis_win_rate"]
                else:
                    eval_res = evaluate_matchup(opp_fn, eval_agent, num_games=games_per_opp)
                    wr = eval_res["soviet_wins"] / max(1, eval_res["games"])
                pool_eval[opp_name] = round(wr * 100, 1)
                pool_win_rates.append(wr)
                clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', opp_name)
                writer.add_scalar(f"eval/{args.train_side}_vs_{clean_name}", wr, global_step)

            last_win_rate = float(np.mean(pool_win_rates)) if pool_win_rates else 0.0
            writer.add_scalar(f"eval/{args.train_side}_win_rate_pool_avg", last_win_rate, global_step)
            eval_summary = " | ".join([f"{n}: {w}%" for n, w in pool_eval.items()])
            print(f"--- Eval @ step {global_step}: {side_label} Win Rate Pool Avg: {last_win_rate*100:.1f}% ({eval_summary}) ---")

        # Write live progress to status file
        resumed_name = os.path.basename(args.resume_checkpoint) if args.resume_checkpoint else None
        status_data = {
            "status": "training",
            "step": global_step,
            "total_steps": args.total_timesteps,
            "progress": round((global_step / args.total_timesteps) * 100, 1),
            "sps": int(global_step / max(1, time.time() - start_time)),
            "policy_loss": round(pg_loss.item(), 4),
            "value_loss": round(v_loss.item(), 4),
            "win_rate": round(last_win_rate * 100, 1),
            "pool_eval": pool_eval if 'pool_eval' in locals() else {},
            "opponents": [name for name, _ in opponent_pool] if opponent_pool else [args.opponent],
            "elapsed": round(time.time() - start_time, 1),
            "device": str(device),
            "train_side": args.train_side,
            "opponent": args.opponent,
            "opponent_checkpoint": os.path.basename(args.opponent_checkpoint) if args.opponent_checkpoint else None,
            "eval_opponent_name": eval_opp_name,
            "resumed_from": resumed_name,
            "num_envs": args.num_envs,
            "reward": round(ema_reward, 4),
            "reward_history": reward_history,
        }
        write_status(args.status_file, status_data)

    # Save final model
    total_completed_steps = prior_steps + global_step
    if args.model_name and args.model_name.strip():
        model_fname = os.path.basename(args.model_name.strip())
        if not model_fname.endswith(".pt"):
            model_fname = f"{model_fname}.pt"
        save_path = os.path.join(args.save_dir, model_fname)
    else:
        save_path = os.path.join(args.save_dir, f"{args.exp_name}_final.pt")
    torch.save({
        "model_state_dict": agent.state_dict(),
        "total_steps": total_completed_steps,
    }, save_path)
    print(f"Model saved to {save_path} (total steps: {total_completed_steps})")

    # Update checkpoints_meta.json
    try:
        meta_path = os.path.join(args.save_dir, "checkpoints_meta.json")
        meta = {}
        if os.path.isfile(meta_path):
            with open(meta_path, "r") as mf:
                meta = json.load(mf)
        meta[os.path.basename(save_path)] = {
            "total_steps": total_completed_steps,
            "timestamp": int(time.time()),
        }
        with open(meta_path, "w") as mf:
            json.dump(meta, mf, indent=2)
    except Exception as exc:
        print(f"Warning: could not write checkpoints_meta.json: {exc}")

    # Mark completed in status file
    status_data["status"] = "completed"
    status_data["latest_checkpoint"] = save_path
    write_status(args.status_file, status_data)

    writer.close()


if __name__ == "__main__":
    train()
