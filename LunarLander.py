import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import deque
import csv
import json
import random

def set_global_seed(seed, deterministic_torch: bool = True):
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def make_episode_rng(seed):
    if seed is None:
        return None
    return np.random.default_rng(seed)


def capture_rng_state(episode_rng):
    state = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "episode_rng_state": None if episode_rng is None else episode_rng.bit_generator.state,
    }

    if torch.cuda.is_available():
        state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

    return state


def restore_rng_state(checkpoint, seed):
    has_rng_state = (
        "python_random_state" in checkpoint
        and "numpy_random_state" in checkpoint
        and "torch_rng_state" in checkpoint
        and "episode_rng_state" in checkpoint
    )

    if has_rng_state:
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])

        if torch.cuda.is_available() and "torch_cuda_rng_state_all" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng_state_all"])

        if checkpoint["episode_rng_state"] is None:
            episode_rng = None
        else:
            episode_rng = np.random.default_rng()
            episode_rng.bit_generator.state = checkpoint["episode_rng_state"]

        return episode_rng

    # fallback for old checkpoints
    set_global_seed(seed)
    return make_episode_rng(seed)


def add_rng_state_to_checkpoint(checkpoint_data, episode_rng):
    checkpoint_data.update(capture_rng_state(episode_rng))
    return checkpoint_data

def next_episode_seed(rng):
    if rng is None:
        return None
    return int(rng.integers(0, 2**32 - 1, dtype=np.uint32))

def summarize_curve(values, window=100):
    arr = np.asarray(values, dtype=np.float64)

    out = {
        "episodes": int(len(arr)),
        "mean_all": float(arr.mean()) if len(arr) else None,
        "std_all": float(arr.std()) if len(arr) else None,
        "min_all": float(arr.min()) if len(arr) else None,
        "max_all": float(arr.max()) if len(arr) else None,
    }

    if len(arr) >= window:
        ma = np.convolve(arr, np.ones(window) / window, mode="valid")
        out["mean_last_100"] = float(arr[-100:].mean())
        out["best_100avg"] = float(ma.max())
    else:
        out["mean_last_100"] = None
        out["best_100avg"] = None

    return out

def save_seed_sweep_results(save_dir, summary_rows, full_results, stem="seed_sweep"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # full curves / losses / metadata
    torch.save(full_results, save_dir / f"{stem}_full.pt")

    # json summary
    with open(save_dir / f"{stem}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    # csv summary
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(save_dir / f"{stem}_summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)


# REINFORCE Policy Gradient Algorithm
class PolicyNetwork(nn.Module):
    def __init__(self, size: int, depth: int):
        super().__init__()
        layers = []

        self.size = size
        self.depth = depth

        if depth == 0:
            self.net = nn.Sequential(nn.Linear(8, 4))
        else:
            layers.append(nn.Linear(8, size))
            layers.append(nn.ReLU())
            for i in range(1, depth):
                layers.append(nn.Linear(size, size))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(size, 4))
            self.net = nn.Sequential(*layers)
        
    def forward(self, state):
        return self.net(state)


def basic_policy(obs): # Naive policy: use side engines to correct horizontal drift.
    if (obs[0] > 0.1):
        return 1
    if (obs[0] < -0.1):
        return 3
    return 0



def choose_action(model, obs): #Action with no eval for REINFORCE
    state = torch.as_tensor(obs, dtype=torch.float32)
    logit = model(state)
    dist = torch.distributions.Categorical(logits=logit)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return int(action.item()), log_prob


def choose_greedy_action_reinforce(model, obs): # Choose greedy for REINFORCE for testing
    with torch.inference_mode():
        state = torch.as_tensor(obs, dtype=torch.float32)
        logits = model(state)
        action = torch.argmax(logits).item()
    return int(action)

def choose_action_and_evaluate(model, obs): #Action with eval for AC
    state = torch.as_tensor(obs, dtype=torch.float32)
    logit, state_value = model(state)
    dist = torch.distributions.Categorical(logits=logit)
    entropy = dist.entropy()
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return int(action.item()), log_prob, state_value, entropy


def choose_greedy_action_ac(model, obs): # Choose greedy for AC for testing
    with torch.inference_mode():
        state = torch.as_tensor(obs, dtype=torch.float32)
        logits, state_value = model(state)
        action = torch.argmax(logits).item()
    return int(action), state_value.item()

def compute_returns(rewards, discount_factor):
    returns = rewards[:]
    for step in range(len(returns) - 1, 0, -1):
        returns[step -1] += returns[step] * discount_factor
        
    return torch.tensor(returns)

def run_episode(model, env, seed=None): #Offline episode log prob and reward collection for REINFORCE
    log_probs, rewards = [], []
    obs, info = env.reset(seed=seed)
    while True:
        action, log_prob = choose_action(model, obs)
        obs, reward, done, truncated, _info = env.step(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        if (done or truncated):
            return log_probs, rewards

def get_reinforce_path(discount_factor, size, depth, lr, seed=None):
    seed_part = "" if seed is None else f"_seed{seed}"
    return Path("reinforce_models") / Path(
        f"reinforce_df{discount_factor}"
        f"_size{size}"
        f"_depth{depth}"
        f"_lr{lr:0.6f}"
        f"{seed_part}"
    )

def train_reinforce(model, optimizer, env, n_episodes, discount_factor, resume=True, seed=None):
    model.train()
    reinforce_dir_path = get_reinforce_path(
        discount_factor,
        model.size,
        model.depth,
        optimizer.param_groups[0]['lr'],
        seed=seed,
    )
    reinforce_dir_path.mkdir(exist_ok=True, parents=True)

    latest_reinforce_path = reinforce_dir_path / Path("latest_reinforce.pt")
    best_reinforce_path = reinforce_dir_path / Path("best_reinforce.pt")

    start_episode = 0
    totals = []
    best_avg = -float("inf")
    set_global_seed(seed)
    
    checkpoint = None
    if resume and latest_reinforce_path.exists():
        checkpoint = torch.load(latest_reinforce_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        totals = checkpoint["totals"]
        best_avg = checkpoint["best_avg"]
        start_episode = checkpoint["episode"] + 1

    episode_rng = restore_rng_state(checkpoint, seed) if checkpoint is not None else make_episode_rng(seed)

    for episode in range(start_episode, n_episodes):
        episode_seed = next_episode_seed(episode_rng)
        log_probs, rewards = run_episode(model, env, seed=episode_seed)
        returns = compute_returns(rewards, discount_factor)

        totals.append(sum(rewards))

        std_returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-7)
        losses = [-logp * rt for logp, rt in zip(log_probs, std_returns)]
        loss = torch.stack(losses, dim=0).sum()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        if len(totals) >= 100:
            avg100 = np.mean(totals[-100:])
            if avg100 > best_avg:
                best_avg = avg100
                best_ckpt = {
                    "episode": episode,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "totals": totals,
                    "best_avg": best_avg,
                    "model": "reinforce",
                    "size": model.size,
                    "depth": model.depth,
                    "seed": seed,
                }
                add_rng_state_to_checkpoint(best_ckpt, episode_rng)
                torch.save(best_ckpt, best_reinforce_path)

        latest_ckpt = {
            "episode": episode,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "totals": totals,
            "best_avg": best_avg,
            "model": "reinforce",
            "size": model.size,
            "depth": model.depth,
            "seed": seed,
        }
        add_rng_state_to_checkpoint(latest_ckpt, episode_rng)
        torch.save(latest_ckpt, latest_reinforce_path)

        print(f"\rEpisode {episode + 1}, Reward: {sum(rewards):.2f}", end=" ")

    model.eval()
    return totals
    
class ActorCritic(nn.Module):
    def __init__(self, size: int, depth: int):
        super().__init__()
        layers = []

        self.size = size
        self.depth = depth

        if depth == 0:
            self.body = nn.Identity()
            body_out = 8
        else:
            layers.append(nn.Linear(8, size))
            layers.append(nn.ReLU())
            for i in range(1, depth):
                layers.append(nn.Linear(size, size))
                layers.append(nn.ReLU())
            self.body = nn.Sequential(*layers)
            body_out = size

        self.actor_head = nn.Linear(body_out, 4)
        self.critic_head = nn.Linear(body_out, 1)

    def forward(self, state):
        features = self.body(state)
        return self.actor_head(features), self.critic_head(features).squeeze(-1)
    



def ac_training_step(model, optimizer, criterion, state_value, target_value, log_prob, entropy, critic_weight=0.5, entropy_weight=0.0005):
    td_error = target_value - state_value
    actor_loss = -log_prob * td_error.detach() - entropy * entropy_weight
    critic_loss = criterion(state_value, target_value)
    loss = actor_loss + critic_weight * critic_loss
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()



def evaluate_given_action(model, obs, action):
    state = torch.as_tensor(obs, dtype=torch.float32)
    logits, state_value = model(state)
    dist = torch.distributions.Categorical(logits=logits)

    action_tensor = torch.tensor(action, dtype=torch.int64)
    log_prob = dist.log_prob(action_tensor)
    entropy = dist.entropy()

    return log_prob, state_value, entropy

def run_episode_and_train_ac(
    model,
    optimizer,
    criterion,
    env,
    discount_factor,
    critic_weight,
    entropy_weight,
    seed=None,
):
    obs, _ = env.reset(seed=seed)
    total_rewards = 0.0

    while True:
        action, log_prob, state_value, entropy = choose_action_and_evaluate(model, obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        total_rewards += reward

        device = state_value.device
        reward_t = torch.tensor(reward, dtype=torch.float32, device=device)

        with torch.no_grad():
            if terminated or truncated:
                target_value = reward_t
            else:
                next_state = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
                _, next_state_value = model(next_state)
                target_value = reward_t + discount_factor * next_state_value

        td_error = target_value - state_value

        actor_loss = -(log_prob * td_error.detach()) - entropy_weight * entropy
        critic_loss = criterion(state_value, target_value)
        loss = actor_loss + critic_weight * critic_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        if terminated or truncated:
            return total_rewards

        obs = next_obs

def linear_anneal(start_value, end_value, current_step, anneal_steps):
    if anneal_steps <= 0:
        return start_value

    progress = min(current_step / anneal_steps, 1.0)
    return start_value + progress * (end_value - start_value)

def get_ac_path(discount_factor, critic_weight, size, depth, lr, entropy_weight_start, entropy_weight_end, entropy_anneal_episodes, seed=None):
    seed_part = "" if seed is None else f"_seed{seed}"
    return Path("ac_models") / Path(
        f"ac_df{discount_factor}"
        f"_cw{critic_weight:0.3f}"
        f"_size{size}"
        f"_depth{depth}"
        f"_lr{lr:0.6f}"
        f"_ews{entropy_weight_start:0.6f}"
        f"_ewe{entropy_weight_end:0.6f}"
        f"_ewa{entropy_anneal_episodes}"
        f"{seed_part}"
    )

def train_actor_critic(model, optimizer, criterion, env, n_episodes=400, discount_factor=0.95, critic_weight=0.3, entropy_weight_start=0.001, entropy_weight_end=0.0001, entropy_anneal_episodes=400, resume=True, seed=None):
    totals = []
    best_avg = -float("inf")

    ac_dir_path = get_ac_path(
        discount_factor,
        critic_weight,
        model.size,
        model.depth,
        optimizer.param_groups[0]['lr'],
        entropy_weight_start,
        entropy_weight_end,
        entropy_anneal_episodes,
        seed=seed,
    )
    ac_dir_path.mkdir(exist_ok=True, parents=True)

    latest_ac_path = ac_dir_path / Path("latest_actor_critic.pt")
    best_ac_path = ac_dir_path / Path("best_actor_critic.pt")

    start_episode = 0

    set_global_seed(seed)
    checkpoint = None
    if resume and latest_ac_path.exists():
        checkpoint = torch.load(latest_ac_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        totals = checkpoint["totals"]
        best_avg = checkpoint["best_avg"]
        start_episode = checkpoint["episode"] + 1

    episode_rng = restore_rng_state(checkpoint, seed) if checkpoint is not None else make_episode_rng(seed)
    

    model.train()
    for episode in range(start_episode, n_episodes):
        episode_seed = next_episode_seed(episode_rng)

        current_entropy_weight = linear_anneal(
            entropy_weight_start,
            entropy_weight_end,
            episode,
            entropy_anneal_episodes,
        )

        total_rewards = run_episode_and_train_ac(
            model,
            optimizer,
            criterion,
            env,
            discount_factor,
            critic_weight,
            current_entropy_weight,
            seed=episode_seed,
        )
        totals.append(total_rewards)

        if len(totals) >= 100:
            avg100 = np.mean(totals[-100:])
            if avg100 > best_avg:
                best_avg = avg100
                best_ckpt = {
                    "episode": episode,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "totals": totals,
                    "best_avg": best_avg,
                    "model": "ac",
                    "size": model.size,
                    "depth": model.depth,
                    "seed": seed,
                }
                add_rng_state_to_checkpoint(best_ckpt, episode_rng)
                torch.save(best_ckpt, best_ac_path)

        latest_ckpt = {
            "episode": episode,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "totals": totals,
            "best_avg": best_avg,
            "model": "ac",
            "size": model.size,
            "depth": model.depth,
            "seed": seed,
        }
        add_rng_state_to_checkpoint(latest_ckpt, episode_rng)
        torch.save(latest_ckpt, latest_ac_path)

        print(
            f"\rEpisode: {episode + 1}, Reward: {total_rewards:.2f}, "
            f"Avg100: {np.mean(totals[-100:]):.2f}, EW: {current_entropy_weight:.6f}",
            end=""
        )

    model.eval()
    return totals




class DQN(nn.Module):
    def __init__(self, size: int, depth: int):
        super().__init__()
        layers = []

        self.size = size
        self.depth = depth

        if depth == 0:
            self.net = nn.Sequential(nn.Linear(8, 4))
        else:
            layers.append(nn.Linear(8, size))
            layers.append(nn.ReLU())
            for _ in range(1, depth):
                layers.append(nn.Linear(size, size))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(size, 4))
            self.net = nn.Sequential(*layers)

    def forward(self, state):
        return self.net(state)


def choose_action_dqn(model, obs, epsilon):
    if np.random.random() < epsilon:
        return int(np.random.randint(4))

    with torch.inference_mode():
        state = torch.as_tensor(obs, dtype=torch.float32)
        q_values = model(state)
        action = torch.argmax(q_values).item()
    return int(action)

def choose_action_dqn_with_ac(policy_model, ac_model, obs, epsilon, ac_guidance_prob=0.5):
    if np.random.random() < epsilon:
        return int(np.random.randint(4))

    if ac_model is not None and np.random.random() < ac_guidance_prob:
        return choose_greedy_action_ac(ac_model, obs)[0]

    return choose_greedy_action_dqn(policy_model, obs)


def choose_greedy_action_dqn(model, obs):
    with torch.inference_mode():
        state = torch.as_tensor(obs, dtype=torch.float32)
        q_values = model(state)
        action = torch.argmax(q_values).item()
    return int(action)


def sample_batch_dqn(replay_buffer, batch_size):
    batch = random.sample(replay_buffer, batch_size)

    states = torch.as_tensor(
        np.array([item["obs"] for item in batch]),
        dtype=torch.float32,
    )
    actions = torch.as_tensor(
        [item["action"] for item in batch],
        dtype=torch.int64,
    )
    rewards = torch.as_tensor(
        [item["reward"] for item in batch],
        dtype=torch.float32,
    )
    next_states = torch.as_tensor(
        np.array([item["next_obs"] for item in batch]),
        dtype=torch.float32,
    )
    dones = torch.as_tensor(
        [item["terminated"] or item["truncated"] for item in batch],
        dtype=torch.float32,
    )

    return states, actions, rewards, next_states, dones


def dqn_training_step(policy_model, target_model, optimizer, criterion, replay_buffer, batch_size, discount_factor):
    states, actions, rewards, next_states, dones = sample_batch_dqn(
        replay_buffer,
        batch_size,
    )

    q_values = policy_model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = target_model(next_states).max(dim=1).values
        target_values = rewards + discount_factor * next_q_values * (1.0 - dones)

    loss = criterion(q_values, target_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 0.5)
    optimizer.step()

    return loss.item()


def run_episode_and_train_dqn(
    policy_model,
    target_model,
    optimizer,
    criterion,
    env,
    replay_buffer,
    batch_size,
    discount_factor,
    epsilon,
    warmup_steps=1000,
    train_freq=1,
    target_update_freq=250,
    seed=None,
    global_step=0,
    ac_model=None,
    ac_guidance_prob=0.0,
):
    obs, _info = env.reset(seed=seed)
    total_rewards = 0.0
    episode_loss_sum = 0.0
    episode_updates = 0

    while True:
        action = choose_action_dqn_with_ac(
            policy_model,
            ac_model,
            obs,
            epsilon,
            ac_guidance_prob=ac_guidance_prob,
        )

        next_obs, reward, terminated, truncated, _info = env.step(action)
        total_rewards += reward

        replay_buffer.append({
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_obs,
            "terminated": terminated,
            "truncated": truncated,
        })

        global_step += 1

        if len(replay_buffer) >= max(batch_size, warmup_steps) and global_step % train_freq == 0:
            loss_value = dqn_training_step(
                policy_model,
                target_model,
                optimizer,
                criterion,
                replay_buffer,
                batch_size,
                discount_factor,
            )
            episode_loss_sum += loss_value
            episode_updates += 1

        if global_step % target_update_freq == 0:
            target_model.load_state_dict(policy_model.state_dict())

        if terminated or truncated:
            avg_loss = 0.0 if episode_updates == 0 else episode_loss_sum / episode_updates
            return total_rewards, avg_loss, global_step

        obs = next_obs


def get_dqn_path(
    discount_factor,
    size,
    depth,
    lr,
    buffer_size,
    batch_size,
    target_update_freq,
    epsilon_start,
    epsilon_end,
    epsilon_decay_episodes,
    ac_guidance_start=0.0,
    ac_guidance_end=0.0,
    ac_guidance_anneal_episodes=0,
    seed=None,
):
    seed_part = "" if seed is None else f"_seed{seed}"
    return Path("dqn_models")/ Path(
        f"dqn_df{discount_factor}"
        f"_size{size}"
        f"_depth{depth}"
        f"_lr{lr:0.6f}"
        f"_bs{buffer_size}"
        f"_batch{batch_size}"
        f"_tuf{target_update_freq}"
        f"_epss{epsilon_start:0.4f}"
        f"_epse{epsilon_end:0.4f}"
        f"_epsd{epsilon_decay_episodes}"
        f"_acgs{ac_guidance_start:0.2f}"
        f"_acge{ac_guidance_end:0.2f}"
        f"_acga{ac_guidance_anneal_episodes}"
        f"{seed_part}"
    )


def train_dqn(
    policy_model,
    target_model,
    optimizer,
    criterion,
    env,
    n_episodes=400,
    discount_factor=0.99,
    buffer_size=100000,
    batch_size=64,
    target_update_freq=250,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_episodes=300,
    warmup_steps=1000,
    train_freq=1,
    resume=True,
    save_replay_buffer=True,
    ac_model=None,
    ac_guidance_start=0.5,
    ac_guidance_end=0.0,
    ac_guidance_anneal_episodes=150,
    seed=None,
):
    totals = []
    losses = []
    best_avg = -float("inf")
    global_step = 0
    replay_buffer = deque(maxlen=buffer_size)

    dqn_dir_path = get_dqn_path(
        discount_factor,
        policy_model.size,
        policy_model.depth,
        optimizer.param_groups[0]["lr"],
        buffer_size,
        batch_size,
        target_update_freq,
        epsilon_start,
        epsilon_end,
        epsilon_decay_episodes,
        ac_guidance_start=ac_guidance_start,
        ac_guidance_end=ac_guidance_end,
        ac_guidance_anneal_episodes=ac_guidance_anneal_episodes,
        seed=seed,
    )
    dqn_dir_path.mkdir(exist_ok=True, parents=True)

    latest_dqn_path = dqn_dir_path / Path("latest_dqn.pt")
    best_dqn_path = dqn_dir_path / Path("best_dqn.pt")

    start_episode = 0
    checkpoint = None
    set_global_seed(seed)

    if resume and latest_dqn_path.exists():
        checkpoint = torch.load(latest_dqn_path, weights_only=False)
        policy_model.load_state_dict(checkpoint["model_state_dict"])
        target_model.load_state_dict(checkpoint["target_model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        totals = checkpoint["totals"]
        losses = checkpoint.get("losses", [])
        best_avg = checkpoint["best_avg"]
        start_episode = checkpoint["episode"] + 1
        global_step = checkpoint.get("global_step", 0)

        if "replay_buffer" in checkpoint:
            replay_buffer = deque(checkpoint["replay_buffer"], maxlen=buffer_size)
        elif save_replay_buffer:
            raise ValueError("Deterministic DQN resume requires replay_buffer in checkpoint, but it was not found.")
    else:
        target_model.load_state_dict(policy_model.state_dict())

    if ac_model is not None:
        ac_model.eval()

    episode_rng = restore_rng_state(checkpoint, seed) if checkpoint is not None else make_episode_rng(seed)

    policy_model.train()
    target_model.eval()

    for episode in range(start_episode, n_episodes):
        episode_seed = next_episode_seed(episode_rng)

        epsilon = linear_anneal(
            epsilon_start,
            epsilon_end,
            episode,
            epsilon_decay_episodes,
        )

        current_ac_guidance = linear_anneal(
            ac_guidance_start,
            ac_guidance_end,
            episode,
            ac_guidance_anneal_episodes,
        )

        total_rewards, avg_loss, global_step = run_episode_and_train_dqn(
            policy_model,
            target_model,
            optimizer,
            criterion,
            env,
            replay_buffer,
            batch_size,
            discount_factor,
            epsilon,
            warmup_steps=warmup_steps,
            train_freq=train_freq,
            target_update_freq=target_update_freq,
            seed=episode_seed,
            global_step=global_step,
            ac_model=ac_model,
            ac_guidance_prob=current_ac_guidance,
        )

        totals.append(total_rewards)
        losses.append(avg_loss)

        checkpoint_data = {
            "episode": episode,
            "model_state_dict": policy_model.state_dict(),
            "target_model_state_dict": target_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "totals": totals,
            "losses": losses,
            "best_avg": best_avg,
            "global_step": global_step,
            "model": "dqn",
            "size": policy_model.size,
            "depth": policy_model.depth,
            "ac_guidance_start": ac_guidance_start,
            "ac_guidance_end": ac_guidance_end,
            "ac_guidance_anneal_episodes": ac_guidance_anneal_episodes,
            "seed": seed,
            "replay_buffer": list(replay_buffer),
        }

        add_rng_state_to_checkpoint(checkpoint_data, episode_rng)

        if len(totals) >= 100:
            avg100 = np.mean(totals[-100:])
            if avg100 > best_avg:
                best_avg = avg100
                checkpoint_data["best_avg"] = best_avg
                torch.save(checkpoint_data, best_dqn_path)

        torch.save(checkpoint_data, latest_dqn_path)

        print(
            f"\rEpisode: {episode + 1}, Reward: {total_rewards:.2f}, "
            f"Avg100: {np.mean(totals[-100:]):.2f}, "
            f"Eps: {epsilon:.4f}, ACG: {current_ac_guidance:.3f}, Loss: {avg_loss:.4f}",
            end=""
        )

    policy_model.eval()
    return totals, losses

def run_dqn_seed_sweep(
    seeds,
    size,
    depth,
    lr,
    n_episodes=400,
    discount_factor=0.99,
    buffer_size=50000,
    batch_size=64,
    target_update_freq=250,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_episodes=300,
    warmup_steps=1000,
    train_freq=1,
    resume=True,
    ac_model=None,
    ac_guidance_start=0.0,
    ac_guidance_end=0.0,
    ac_guidance_anneal_episodes=1,
    env_name="LunarLander-v2",
    save_root="seed_sweeps",
):
    summary_rows = []
    full_results = {}

    for seed in seeds:
        print(f"\n===== DQN seed {seed} =====")

        set_global_seed(seed)
        env = gym.make(env_name)

        dqn_model = DQN(size, depth)
        target_model = DQN(size, depth)
        optimizer = torch.optim.Adam(dqn_model.parameters(), lr=lr)
        criterion = nn.SmoothL1Loss()

        totals, losses = train_dqn(
            policy_model=dqn_model,
            target_model=target_model,
            optimizer=optimizer,
            criterion=criterion,
            env=env,
            n_episodes=n_episodes,
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_episodes=epsilon_decay_episodes,
            warmup_steps=warmup_steps,
            train_freq=train_freq,
            resume=resume,
            save_replay_buffer=True,
            ac_model=ac_model,
            ac_guidance_start=ac_guidance_start,
            ac_guidance_end=ac_guidance_end,
            ac_guidance_anneal_episodes=ac_guidance_anneal_episodes,
            seed=seed,
        )

        run_summary = summarize_curve(totals, window=100)
        run_summary.update({
            "seed": seed,
            "model": "dqn",
            "size": size,
            "depth": depth,
            "lr": lr,
            "discount_factor": discount_factor,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "target_update_freq": target_update_freq,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "epsilon_decay_episodes": epsilon_decay_episodes,
            "ac_guidance_start": ac_guidance_start,
            "ac_guidance_end": ac_guidance_end,
            "ac_guidance_anneal_episodes": ac_guidance_anneal_episodes,
            "final_loss": float(losses[-1]) if len(losses) else None,
            "mean_loss_last_50": float(np.mean(losses[-50:])) if len(losses) >= 50 else (float(np.mean(losses)) if len(losses) else None),
        })
        summary_rows.append(run_summary)

        full_results[seed] = {
            "totals": totals,
            "losses": losses,
        }

        env.close()

    save_dir = Path(save_root) / (
        f"dqn_size{size}_depth{depth}_lr{lr:0.6f}"
        f"_df{discount_factor}_batch{batch_size}"
    )
    save_seed_sweep_results(
        save_dir=save_dir,
        summary_rows=summary_rows,
        full_results=full_results,
        stem="dqn_seed_sweep",
    )

    return summary_rows, full_results


def run_ac_seed_sweep(
    seeds,
    size,
    depth,
    lr,
    n_episodes=400,
    discount_factor=0.95,
    critic_weight=0.3,
    entropy_weight_start=0.001,
    entropy_weight_end=0.0001,
    entropy_anneal_episodes=400,
    resume=True,
    env_name="LunarLander-v2",
    save_root="seed_sweeps",
):
    summary_rows = []
    full_results = {}

    for seed in seeds:
        print(f"\n===== AC seed {seed} =====")

        set_global_seed(seed)
        env = gym.make(env_name)

        ac_model = ActorCritic(size, depth)
        optimizer = torch.optim.Adam(ac_model.parameters(), lr=lr)
        criterion = nn.SmoothL1Loss()

        totals = train_actor_critic(
            model=ac_model,
            optimizer=optimizer,
            criterion=criterion,
            env=env,
            n_episodes=n_episodes,
            discount_factor=discount_factor,
            critic_weight=critic_weight,
            entropy_weight_start=entropy_weight_start,
            entropy_weight_end=entropy_weight_end,
            entropy_anneal_episodes=entropy_anneal_episodes,
            resume=resume,
            seed=seed,
        )

        run_summary = summarize_curve(totals, window=100)
        run_summary.update({
            "seed": seed,
            "model": "actor_critic",
            "size": size,
            "depth": depth,
            "lr": lr,
            "discount_factor": discount_factor,
            "critic_weight": critic_weight,
            "entropy_weight_start": entropy_weight_start,
            "entropy_weight_end": entropy_weight_end,
            "entropy_anneal_episodes": entropy_anneal_episodes,
        })

        summary_rows.append(run_summary)
        full_results[seed] = {
            "totals": totals,
        }

        env.close()

    save_dir = Path(save_root) / (
        f"ac_size{size}_depth{depth}_lr{lr:0.6f}"
        f"_df{discount_factor}"
        f"_cw{critic_weight:0.3f}"
        f"_ews{entropy_weight_start:0.6f}"
        f"_ewe{entropy_weight_end:0.6f}"
        f"_ewa{entropy_anneal_episodes}"
    )

    save_seed_sweep_results(
        save_dir=save_dir,
        summary_rows=summary_rows,
        full_results=full_results,
        stem="ac_seed_sweep",
    )

    return summary_rows, full_results


def run_reinforce_seed_sweep(
    seeds,
    size,
    depth,
    lr,
    n_episodes=400,
    discount_factor=0.99,
    resume=True,
    env_name="LunarLander-v2",
    save_root="seed_sweeps",
):
    summary_rows = []
    full_results = {}

    for seed in seeds:
        print(f"\n===== REINFORCE seed {seed} =====")

        set_global_seed(seed)
        env = gym.make(env_name)

        model = PolicyNetwork(size, depth)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        totals = train_reinforce(
            model=model,
            optimizer=optimizer,
            env=env,
            n_episodes=n_episodes,
            discount_factor=discount_factor,
            resume=resume,
            seed=seed,
        )

        run_summary = summarize_curve(totals, window=100)
        run_summary.update({
            "seed": seed,
            "model": "reinforce",
            "size": size,
            "depth": depth,
            "lr": lr,
            "discount_factor": discount_factor,
        })

        summary_rows.append(run_summary)
        full_results[seed] = {
            "totals": totals,
        }

        env.close()

    save_dir = Path(save_root) / (
        f"reinforce_size{size}_depth{depth}_lr{lr:0.6f}"
        f"_df{discount_factor}"
    )

    save_seed_sweep_results(
        save_dir=save_dir,
        summary_rows=summary_rows,
        full_results=full_results,
        stem="reinforce_seed_sweep",
    )

    return summary_rows, full_results


def load_model_for_eval(path):
    checkpoint = torch.load(path, weights_only=False)
    model_type = checkpoint["model"]
    size = checkpoint["size"]
    depth = checkpoint["depth"]
    
    if model_type == "reinforce":
        model = PolicyNetwork(size, depth)
    elif model_type == "ac":
        model = ActorCritic(size, depth)
    elif model_type == "dqn":
        model = DQN(size, depth)
        

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, model_type



# If the script is ran by itself, use it to test a model
if __name__ == "__main__": 
    import argparse

    parser = argparse.ArgumentParser(description="Helpful utilities for Lunar Lander")

    parser.add_argument("--checkpoint", type=str, default=None, help="Path to saved checkpoint (.pt), if this is invalid or None, naive agent will run instead")
    parser.add_argument(
            "--episodes",
            type=int,
            default=5,
            help="Number of test episodes to run.",
        )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed for evaluation.",
    )

    args = parser.parse_args()

    if args.checkpoint is not None and Path(args.checkpoint).exists():
        try:
            model, model_type = load_model_for_eval(Path(args.checkpoint))
        except Exception as e:
            model_type = "naive"
            print(e)
            print("Falling back to naive, could not load checkpoint")
    else:
        model_type = "naive"
        print("Falling back to naive, could not load checkpoint")


        
        

    env = gym.make("LunarLander-v2", render_mode="human")
    
    


    final_totals = []    
    for episode in range(args.episodes):
        episode_seed = None if args.seed is None else args.seed + episode
        obs,info = env.reset(seed=episode_seed)
        totals = []
        while True:
            if (model_type == "naive"):
                action = basic_policy(obs)
            elif (model_type == "reinforce"):
                action = choose_greedy_action_reinforce(model, obs)
            elif (model_type == "ac"):
                action = choose_greedy_action_ac(model, obs)[0]
            elif (model_type == "dqn"):
                action = choose_greedy_action_dqn(model, obs)
            obs, reward, done, truncated, info = env.step(action)
            totals.append(reward)
            if done or truncated:
                break
        mean = sum(totals)
        print(f"Episode rewards total: {mean}")
        final_totals.append(mean)
    

    print("Final: ")
    print(np.mean(final_totals),np.std(final_totals),min(final_totals),max(final_totals))

    
    env.close()
