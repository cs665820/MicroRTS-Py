#!/usr/bin/env python3
import argparse
import os
import random
import subprocess
import time
from distutils.util import strtobool
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from gym.spaces import MultiDiscrete
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, VecVideoRecorder
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv


def parse_args():
    parser = argparse.ArgumentParser()
    # ─── General experiment args ────────────────────────────────────────────────────
    parser.add_argument('--exp-name', type=str,
        default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="MicroRTSGridModeVecEnv",
        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=50000000,
        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic',
        type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda',
        type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode',
        type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video',
        type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='whether to capture videos of the agent performances')
    parser.add_argument('--wandb-project-name', type=str, default="gym-microrts",
        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
        help="the entity (team) of wandb's project")

    # ─── Algorithm-specific arguments ───────────────────────────────────────────────
    parser.add_argument('--reward-weight', type=float, nargs='+',
        default=[10.0,1.0,1.0,0.2,1.0,4.0],
        help='weights for microrts raw rewards')
    parser.add_argument('--partial-obs',
        type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, the game will have partial observability')
    parser.add_argument('--n-minibatch', type=int, default=4,
        help='the number of mini batch')
    parser.add_argument('--num-bot-envs', type=int, default=0,
        help='the number of bot game environments')
    parser.add_argument('--num-selfplay-envs', type=int, default=24,
        help='the number of self play envs')
    parser.add_argument('--num-steps', type=int, default=256,
        help='the number of steps per game environment')
    parser.add_argument('--gamma', type=float, default=0.99,
        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
        help='the lambda for the generalized advantage estimation')
    parser.add_argument('--ent-coef', type=float, default=0.01,
        help='coefficient of the entropy bonus')
    parser.add_argument('--vf-coef', type=float, default=0.5,
        help='coefficient of the value function loss')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
        help='maximum norm for gradient clipping')
    parser.add_argument('--clip-coef', type=float, default=0.1,
        help='PPO surrogate clipping coefficient (unused)')
    parser.add_argument('--update-epochs', type=int, default=4,
        help='number of epochs to update policy/value per rollout (unused)')
    parser.add_argument('--kle-stop',
        type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, early stop on target‑KL (unused)')
    parser.add_argument('--kle-rollback',
        type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, roll back update if KL exceeds target (unused)')
    parser.add_argument('--target-kl', type=float, default=0.03,
        help='target KL divergence for early stopping/rollback (unused)')
    parser.add_argument('--gae',
        type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='toggle using GAE for advantage computation')
    parser.add_argument('--norm-adv',
        type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='toggle normalization of advantages')
    parser.add_argument('--anneal-lr',
        type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='toggle learning rate annealing for policy/value nets')
    parser.add_argument('--clip-vloss',
        type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='toggle value‑loss clipping')
    parser.add_argument('--kl-coef', type=float, default=1.0,
        help='coefficient for the KL penalty in the policy loss')
    parser.add_argument('--num-models', type=int, default=100,
        help='the number of model checkpoints to keep')
    parser.add_argument('--max-eval-workers', type=int, default=4,
        help='max number of parallel evaluation workers')
    parser.add_argument('--train-maps', nargs='+',
        default=["maps/16x16/basesWorkers16x16A.xml"],
        help='list of maps used during training')
    parser.add_argument('--eval-maps', nargs='+',
        default=["maps/16x16/basesWorkers16x16A.xml"],
        help='list of maps used during evaluation')

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())
    args.num_envs       = args.num_selfplay_envs + args.num_bot_envs
    args.batch_size     = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.n_minibatch)
    args.num_updates    = args.total_timesteps // args.batch_size
    args.save_frequency = max(1, int(args.num_updates // args.num_models))
    return args


class MicroRTSStatsRecorder(VecEnvWrapper):
    def __init__(self, env, gamma=0.99) -> None:
        super().__init__(env)
        self.gamma = gamma

    def reset(self):
        obs = self.venv.reset()
        self.raw_rewards = [[] for _ in range(self.num_envs)]
        self.ts = np.zeros(self.num_envs, dtype=np.float32)
        self.raw_discount_rewards = [[] for _ in range(self.num_envs)]
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        newinfos = list(infos)
        for i in range(len(dones)):
            self.raw_rewards[i] += [infos[i]["raw_rewards"]]
            self.raw_discount_rewards[i] += [
                (self.gamma ** self.ts[i]) *
                np.concatenate((infos[i]["raw_rewards"],
                                infos[i]["raw_rewards"].sum()),
                               axis=None)
            ]
            self.ts[i] += 1
            if dones[i]:
                info = infos[i].copy()
                raw_returns = np.array(self.raw_rewards[i]).sum(0)
                raw_names = [str(rf) for rf in self.rfs]
                raw_discount_returns = np.array(self.raw_discount_rewards[i]).sum(0)
                raw_discount_names = ["discounted_" + str(rf)
                                      for rf in self.rfs] + ["discounted"]
                info["microrts_stats"] = dict(zip(raw_names, raw_returns))
                info["microrts_stats"].update(
                    dict(zip(raw_discount_names, raw_discount_returns)))
                self.raw_rewards[i] = []
                self.raw_discount_rewards[i] = []
                self.ts[i] = 0
                newinfos[i] = info
        return obs, rews, dones, newinfos


class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None,
                 validate_args=None, masks=[], mask_value=None):
        logits = torch.where(masks.bool(), logits, mask_value)
        super().__init__(probs, logits, validate_args)


class Transpose(nn.Module):
    def __init__(self, permutation):
        super().__init__()
        self.permutation = permutation

    def forward(self, x):
        return x.permute(self.permutation)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, mapsize=16 * 16):
        super().__init__()
        self.mapsize = mapsize
        self.envs = envs
        h, w, c = envs.observation_space.shape
        self.encoder = nn.Sequential(
            Transpose((0, 3, 1, 2)),
            layer_init(nn.Conv2d(c, 32, 3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            layer_init(nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1,
                                         output_padding=1)),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(32, 78, 3, stride=2, padding=1,
                                         output_padding=1)),
            Transpose((0, 2, 3, 1)),
        )
        self.critic = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(64 * 4 * 4, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=1),
        )
        self.register_buffer("mask_value", torch.tensor(-1e8))

    def get_action(self, obs, invalid_action_masks):
        hidden = self.encoder(obs)
        logits = self.actor(hidden)
        grid_logits = logits.reshape(-1,
            self.envs.action_plane_space.nvec.sum())
        split_logits = torch.split(
        grid_logits, self.envs.action_plane_space.nvec.tolist(), dim=1)
        invalid = invalid_action_masks.view(
            -1, invalid_action_masks.shape[-1])
        split_invalid = torch.split(
            invalid, self.envs.action_plane_space.nvec.tolist(), dim=1)
        cats = [CategoricalMasked(logits=lg, masks=msk,
                                  mask_value=self.mask_value)
                for lg, msk in zip(split_logits, split_invalid)]
        action = torch.stack([c.sample() for c in cats])
        logprob = torch.stack([c.log_prob(a) for a, c in zip(action, cats)])
        entropy = torch.stack([c.entropy() for c in cats])
        num_pred = len(self.envs.action_plane_space.nvec)
        logprob = logprob.T.view(-1, self.mapsize, num_pred)
        entropy = entropy.T.view(-1, self.mapsize, num_pred)
        action = action.T.view(-1, self.mapsize, num_pred)
        return action

    def get_action_and_value(self, x, action=None,
                             invalid_action_masks=None,
                             envs=None, device=None):
        hidden = self.encoder(x)
        logits = self.actor(hidden)
        grid_logits = logits.reshape(-1,
            envs.action_plane_space.nvec.sum())
        split_logits = torch.split(
            grid_logits, envs.action_plane_space.nvec.tolist(), dim=1)

        if action is None:
            invalid = invalid_action_masks.view(
                -1, invalid_action_masks.shape[-1])
            split_invalid = torch.split(
                invalid, envs.action_plane_space.nvec.tolist(), dim=1)
            cats = [CategoricalMasked(logits=lg, masks=msk,
                                      mask_value=self.mask_value)
                    for lg, msk in zip(split_logits, split_invalid)]
            action = torch.stack([c.sample() for c in cats])
        else:
            invalid = invalid_action_masks.view(
                -1, invalid_action_masks.shape[-1])
            action = action.view(-1, action.shape[-1]).T
            split_invalid = torch.split(
                invalid, envs.action_plane_space.nvec.tolist(), dim=1)
            cats = [CategoricalMasked(logits=lg, masks=msk,
                                      mask_value=self.mask_value)
                    for lg, msk in zip(split_logits, split_invalid)]
        logprob = torch.stack([c.log_prob(a) for a, c in zip(action, cats)])
        entropy = torch.stack([c.entropy() for c in cats])
        num_pred = len(envs.action_plane_space.nvec)
        logprob = logprob.T.view(-1, self.mapsize, num_pred)
        entropy = entropy.T.view(-1, self.mapsize, num_pred)
        action = action.T.view(-1, self.mapsize, num_pred)
        return action, logprob.sum(1).sum(1), entropy.sum(1).sum(1), invalid, \
               self.critic(hidden)

    def get_value(self, x):
        return self.critic(self.encoder(x))


def run_evaluation(model_path: str, output_path: str, eval_maps: List[str]):
    args = [
        "python", "league.py", "--evals", model_path, "--update-db", "false",
        "--cuda", "false", "--output-path", output_path,
        "--model-type", "ppo_gridnet", "--maps", *eval_maps
    ]
    fd = subprocess.Popen(args)
    print(f"Evaluating {model_path}")
    assert fd.wait() == 0
    return model_path, output_path


class TrueskillWriter:
    def __init__(self, prod_mode, writer,
                 league_path: str, league_step_path: str):
        self.prod_mode = prod_mode
        self.writer = writer
        self.trueskill_df = pd.read_csv(league_path)
        self.trueskill_step_df = pd.read_csv(league_step_path)
        self.trueskill_step_df["type"] = self.trueskill_step_df["name"]
        self.trueskill_step_df["step"] = 0
        self.preset_trueskill_step_df = self.trueskill_step_df.copy()

    def on_evaluation_done(self, future):
        if future.cancelled():
            return
        model_path, output_path = future.result()
        league = pd.read_csv(output_path, index_col="name")
        assert model_path in league.index
        step = int(model_path.split("/")[-1][:-3])
        self.writer.add_scalar("charts/trueskill",
            league.loc[model_path]["trueskill"], step)
        if self.prod_mode:
            data = {
                "name": league.loc[model_path].name,
                "mu": league.loc[model_path]["mu"],
                "sigma": league.loc[model_path]["sigma"],
                "trueskill": league.loc[model_path]["trueskill"]
            }
            self.trueskill_df = self.trueskill_df.append(
                data, ignore_index=True)
            import wandb
            wandb.log({"trueskill": wandb.Table(dataframe=self.trueskill_df)})
            data["type"] = "training"
            data["step"] = step
            self.trueskill_step_df = self.trueskill_step_df.append(
                data, ignore_index=True)
            clone = self.preset_trueskill_step_df.copy()
            clone["step"] = step
            self.trueskill_step_df = self.trueskill_step_df.append(
                clone, ignore_index=True)
            wandb.log({"trueskill_step":
                       wandb.Table(dataframe=self.trueskill_step_df)})


if __name__ == "__main__":
    args = parse_args()
    print(f"Save frequency: {args.save_frequency}")

    # Setup environment & logging
    experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}\
__{int(time.time())}"
    if args.prod_mode:
        import wandb
        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            sync_tensorboard=False,
            name=experiment_name,
            monitor_gym=True,
            save_code=True,
        )
        wandb.tensorboard.patch(save=False)

    writer = SummaryWriter(f"runs/{experiment_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" +
        "\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])
    )

    # Seeding & device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda \
                          else "cpu")
    print(f"Device: {device}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Environment setup
    envs = MicroRTSGridModeVecEnv(
        num_selfplay_envs=args.num_selfplay_envs,
        num_bot_envs=args.num_bot_envs,
        partial_obs=args.partial_obs,
        max_steps=2000,
        render_theme=2,
        ai2s=[microrts_ai.coacAI for _ in \
            range(args.num_bot_envs-6)] +
             [microrts_ai.randomBiasedAI for _ in \
            range(min(args.num_bot_envs,2))] +
             [microrts_ai.lightRushAI for _ in \
            range(min(args.num_bot_envs,2))] +
             [microrts_ai.workerRushAI for _ in \
            range(min(args.num_bot_envs,2))],
        map_paths=[args.train_maps[0]],
        reward_weight=np.array(args.reward_weight),
        cycle_maps=args.train_maps,
    )
    envs = MicroRTSStatsRecorder(envs, args.gamma)
    envs = VecMonitor(envs)
    if args.capture_video:
        envs = VecVideoRecorder(
            envs, f"videos/{experiment_name}",
            record_video_trigger=lambda x: x % 100000 == 0,
            video_length=2000
        )
    assert isinstance(envs.action_space, MultiDiscrete), \
        "only MultiDiscrete action space is supported"

    # Eval pool
    eval_executor = None
    if args.max_eval_workers > 0:
        from concurrent.futures import ThreadPoolExecutor
        eval_executor = ThreadPoolExecutor(
            max_workers=args.max_eval_workers,
            thread_name_prefix="league-eval-"
        )

    # Agent & optimizer
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(),
                           lr=args.learning_rate, eps=1e-5)
    if args.anneal_lr:
        lr = lambda f: f * args.learning_rate

    # Storage buffers
    mapsize = 16 * 16
    action_space_shape = (mapsize,
                          len(envs.action_plane_space.nvec))
    invalid_action_shape = (mapsize,
        envs.action_plane_space.nvec.sum())

    obs     = torch.zeros((args.num_steps, args.num_envs)
                         + envs.observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs)
                         + action_space_shape).to(device)
    logprobs= torch.zeros((args.num_steps, args.num_envs)
                         ).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)
                         ).to(device)
    dones   = torch.zeros((args.num_steps, args.num_envs)
                         ).to(device)
    invalid_action_masks = torch.zeros(
        (args.num_steps, args.num_envs) + invalid_action_shape
    ).to(device)
    values  = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time  = time.time()
    next_obs    = torch.Tensor(envs.reset()).to(device)
    next_done   = torch.zeros(args.num_envs).to(device)

    # Crash & resume logic
    starting_update = 1
    if args.prod_mode and run.resumed:
        starting_update = run.summary.get("charts/update") + 1
        global_step     = starting_update * args.batch_size
        api             = wandb.Api()
        run             = api.run(f"{run.entity}/{run.project}/{run.id}")
        model_file      = run.file("agent.pt")
        model_file.download(f"models/{experiment_name}/")
        agent.load_state_dict(
            torch.load(f"models/{experiment_name}/agent.pt",
                       map_location=device)
        )
        agent.eval()
        print(f"resumed at update {starting_update}")

    print("Model's state_dict:")
    for name, param in agent.state_dict().items():
        print(name, "\t", param.size())
    total_params = sum(p.nelement() for p in agent.parameters())
    print("Model's total parameters:", total_params)

    trueskill_writer = TrueskillWriter(
        args.prod_mode, writer,
        "gym-microrts-static-files/league.csv",
        "gym-microrts-static-files/league.csv"
    )

    # ──────────────────────────────────────────────────────────────────────────────
    #                             Main training loop
    # ──────────────────────────────────────────────────────────────────────────────
    for update in range(starting_update, args.num_updates + 1):
        # optional LR annealing
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / args.num_updates
            optimizer.param_groups[0]["lr"] = lr(frac)

        # rollout
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step]     = next_obs
            dones[step]   = next_done
            with torch.no_grad():
                invalid_action_masks[step] = torch.tensor(
                    envs.get_action_mask()).to(device)
                action, logprob, _, _, vs = \
                    agent.get_action_and_value(
                        next_obs,
                        envs=envs,
                        invalid_action_masks=invalid_action_masks[step],
                        device=device
                    )
                actions[step]   = action
                logprobs[step]  = logprob
                values[step]    = vs.flatten()

            next_obs, rs, ds, infos = envs.step(
                action.cpu().numpy().reshape(envs.num_envs, -1)
            )
            next_obs  = torch.Tensor(next_obs).to(device)
            rewards[step], next_done = torch.Tensor(rs).to(device), \
                                       torch.Tensor(ds).to(device)

            for info in infos:
                if "episode" in info:
                    writer.add_scalar(
                        "charts/episodic_return",
                        info["episode"]["r"], global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_length",
                        info["episode"]["l"], global_step
                    )
                    for k in info["microrts_stats"]:
                        writer.add_scalar(
                            f"charts/episodic_return/{k}",
                            info["microrts_stats"][k], global_step
                        )
                    break

        # calculate returns & advantages via GAE
        returns    = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
        lastgaelam = torch.zeros(args.num_envs, device=device)
        for t in reversed(range(args.num_steps)):
            mask  = 1.0 - dones[t]
            delta = rewards[t] + args.gamma * next_value * mask - values[t]
            advantages[t] = lastgaelam = (
                delta + args.gamma * args.gae_lambda * mask * lastgaelam
            )
            returns[t]    = advantages[t] + values[t]
            next_value    = values[t]

        # flatten
        b_obs        = obs.reshape((-1,) + envs.observation_space.shape)
        b_actions    = actions.reshape((-1,) + action_space_shape)
        b_logprobs   = logprobs.reshape(-1)
        b_returns    = returns.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_masks      = invalid_action_masks.reshape(
            (-1,) + invalid_action_shape)

        # preserve old logprobs for KL penalty
        b_logprobs_old = b_logprobs.clone().detach()

        # recompute under current policy
        _, newlogprob, entropy, _, newvalue = \
            agent.get_action_and_value(
                b_obs,
                action=b_actions.long(),
                invalid_action_masks=b_masks,
                envs=envs,
                device=device
            )
        newvalue = newvalue.view(-1)

        # KL‐penalty policy loss
        kl = (b_logprobs_old - newlogprob).mean()
        policy_loss_plain = -(b_advantages * newlogprob).mean()
        policy_loss       = policy_loss_plain + args.kl_coef * kl

        # critic loss and entropy bonus
        value_loss   = 0.5 * (newvalue - b_returns).pow(2).mean()
        entropy_loss = - args.ent_coef * entropy.mean()

        loss = policy_loss + args.vf_coef * value_loss + entropy_loss

        # backward & step
        optimizer.zero_grad()
        loss.backward()
        for name, param in agent.named_parameters():
         if param.grad is not None:
             print(f"{name:30s} grad_norm={param.grad.norm().item():.4f}")
        nn.utils.clip_grad_norm_(
            agent.parameters(), args.max_grad_norm)
        optimizer.step()

        # logging
        if args.prod_mode:
            import wandb
            wandb.log({
                "policy_loss":    policy_loss.item(),
                "value_loss":     value_loss.item(),
                "entropy":        entropy.mean().item(),
                "kl":             kl.item(),
                "learning_rate":  optimizer.param_groups[0]["lr"],
                "sps":            int(global_step / (time.time() - start_time))
            }, step=global_step)

        if (update - 1) % args.save_frequency == 0:
            os.makedirs(f"models/{experiment_name}", exist_ok=True)
            torch.save(agent.state_dict(),
                       f"models/{experiment_name}/agent.pt")
            torch.save(agent.state_dict(),
                       f"models/{experiment_name}/{global_step}.pt")
            if args.prod_mode:
                wandb.save(
                    f"models/{experiment_name}/agent.pt",
                    base_path=f"models/{experiment_name}",
                    policy="now"
                )
            if eval_executor is not None:
                future = eval_executor.submit(
                    run_evaluation,
                    f"models/{experiment_name}/{global_step}.pt",
                    f"runs/{experiment_name}/{global_step}.csv",
                    args.eval_maps,
                )
                future.add_done_callback(
                    trueskill_writer.on_evaluation_done)

        # TensorBoard scalars
        writer.add_scalar("charts/learning_rate",
                          optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/update", update, global_step)
        writer.add_scalar("losses/value_loss",
                          value_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss",
                          policy_loss_plain.item(), global_step)
        writer.add_scalar("losses/entropy",
                          entropy.detach().mean().item(), global_step)
        writer.add_scalar("charts/sps",
                          int(global_step / (time.time() - start_time)), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))

    # final clean up
    if eval_executor is not None:
        eval_executor.shutdown(wait=True, cancel_futures=False)
    envs.close()
    writer.close()
