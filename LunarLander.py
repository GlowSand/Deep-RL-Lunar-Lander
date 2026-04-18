import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import deque

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

def get_reinforce_path(discount_factor, size, depth, lr):
    return Path(f"reinforce_df{discount_factor}_size{size}_depth{depth}_lr{lr:0.6f}")

def train_reinforce(model, optimizer, env, n_episodes, discount_factor,  resume=True):
    model.train()
    reinforce_dir_path = get_reinforce_path(discount_factor, model.size, model.depth, optimizer.param_groups[0]['lr'])

    reinforce_dir_path.mkdir(exist_ok=True)
    
    latest_reinforce_path = reinforce_dir_path / Path("latest_reinforce.pt")
    best_reinforce_path = reinforce_dir_path / Path("best_reinforce.pt")

    start_episode = 0
    totals = []
    best_avg = -float("inf")

    
    if (resume and latest_reinforce_path.exists()):
        checkpoint = torch.load(latest_reinforce_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        totals = checkpoint["totals"]
        best_avg = checkpoint["best_avg"]
        start_episode = checkpoint["episode"] + 1
    
    for episode in range(start_episode, n_episodes):
        seed = torch.randint(0, 2**32, size=()).item()
        log_probs, rewards = run_episode(model, env, seed=seed)
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
                torch.save({
                    'episode': episode,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'totals': totals,
                    "best_avg": best_avg,
                    "model": "reinforce",
                    "size": model.size,
                    "depth": model.depth
                }, best_reinforce_path)
        torch.save({
                'episode': episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'totals': totals,
                "best_avg": best_avg,
                "model": "reinforce",
                "size": model.size,
                "depth": model.depth
            }, latest_reinforce_path)
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

def get_n_step_target(model, transition_buffer, discount_factor, n_steps):
    device = next(model.parameters()).device
    items = list(transition_buffer)[:n_steps]

    target_value = torch.tensor(0.0, dtype=torch.float32, device=device)

    for i, item in enumerate(items):
        target_value = target_value + (discount_factor ** i) * item["reward"]

        if item["terminated"] or item["truncated"]:
            return target_value

    with torch.inference_mode():
        next_state = torch.as_tensor(items[-1]["next_obs"], dtype=torch.float32, device=device)
        _, next_state_value = model(next_state)

    target_value = target_value + (discount_factor ** len(items)) * next_state_value
    return target_value

def evaluate_given_action(model, obs, action):
    state = torch.as_tensor(obs, dtype=torch.float32)
    logits, state_value = model(state)
    dist = torch.distributions.Categorical(logits=logits)

    action_tensor = torch.tensor(action, dtype=torch.int64)
    log_prob = dist.log_prob(action_tensor)
    entropy = dist.entropy()

    return log_prob, state_value, entropy

def run_episode_and_train_ac(model, optimizer, criterion, env, discount_factor, critic_weight, entropy_weight, n_steps=5, seed=None):
    obs, _info = env.reset(seed=seed)
    total_rewards = 0.0
    transition_buffer = deque()

    while True:
        action, _log_prob, _state_value, _entropy = choose_action_and_evaluate(model, obs)
        next_obs, reward, terminated, truncated, _info = env.step(action)
        total_rewards += reward

        transition_buffer.append({
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_obs,
            "terminated": terminated,
            "truncated": truncated,
        })

        if len(transition_buffer) >= n_steps:
            target_value = get_n_step_target(model, transition_buffer, discount_factor, n_steps)
            oldest = transition_buffer.popleft()

            log_prob, state_value, entropy = evaluate_given_action(
                model,
                oldest["obs"],
                oldest["action"],
            )

            ac_training_step(
                model,
                optimizer,
                criterion,
                state_value,
                target_value,
                log_prob,
                entropy,
                critic_weight,
                entropy_weight,
            )

        if terminated or truncated:
            while transition_buffer:
                target_value = get_n_step_target(
                    model,
                    transition_buffer,
                    discount_factor,
                    len(transition_buffer),
                )
                oldest = transition_buffer.popleft()

                log_prob, state_value, entropy = evaluate_given_action(
                    model,
                    oldest["obs"],
                    oldest["action"],
                )

                ac_training_step(
                    model,
                    optimizer,
                    criterion,
                    state_value,
                    target_value,
                    log_prob,
                    entropy,
                    critic_weight,
                    entropy_weight,
                )

            return total_rewards

        obs = next_obs

def linear_anneal(start_value, end_value, current_step, anneal_steps):
    if anneal_steps <= 0:
        return start_value

    progress = min(current_step / anneal_steps, 1.0)
    return start_value + progress * (end_value - start_value)

def get_ac_path(discount_factor, critic_weight, size, depth, lr, entropy_weight_start, entropy_weight_end, entropy_anneal_episodes, n_steps):
    return Path(
        f"ac_df{discount_factor}"
        f"_cw{critic_weight:0.3f}"
        f"_size{size}"
        f"_depth{depth}"
        f"_lr{lr:0.6f}"
        f"_ews{entropy_weight_start:0.6f}"
        f"_ewe{entropy_weight_end:0.6f}"
        f"_ewa{entropy_anneal_episodes}"
        f"_ns{n_steps}"
    )

def train_actor_critic(model, optimizer, criterion, env, n_episodes=400, discount_factor=0.95, critic_weight=0.3, entropy_weight_start=0.001, entropy_weight_end=0.0001, entropy_anneal_episodes=400, n_steps=5, resume=True):
    totals = []
    best_avg = -float("inf")

    ac_dir_path = get_ac_path(discount_factor, critic_weight, model.size, model.depth, optimizer.param_groups[0]['lr'], entropy_weight_start, entropy_weight_end, entropy_anneal_episodes, n_steps)
    ac_dir_path.mkdir(exist_ok=True)
    
    latest_ac_path = ac_dir_path / Path("latest_actor_critic.pt")
    best_ac_path = ac_dir_path / Path("best_actor_critic.pt")
    
    start_episode = 0
    if (resume and latest_ac_path.exists()):
        checkpoint = torch.load(latest_ac_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        totals = checkpoint["totals"]
        best_avg = checkpoint["best_avg"]
        start_episode = checkpoint["episode"] + 1
    
    model.train()
    for episode in range(start_episode, n_episodes):
        seed = torch.randint(0, 2**32, size=()).item()
        current_entropy_weight = linear_anneal(
            entropy_weight_start,
            entropy_weight_end,
            episode,
            entropy_anneal_episodes,
        )
        total_rewards = run_episode_and_train_ac(model, optimizer, criterion, env, discount_factor, critic_weight, current_entropy_weight, n_steps, seed=seed)
        totals.append(total_rewards)

        if len(totals) >= 100:
            avg100 = np.mean(totals[-100:])
            if avg100 > best_avg:
                best_avg = avg100
                torch.save({
                    'episode': episode,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'totals': totals,
                    "best_avg": best_avg,
                    "model": "ac",
                    "size": model.size,
                    "depth": model.depth
                }, best_ac_path)
        
        torch.save({
                'episode': episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'totals': totals,
                "best_avg": best_avg,
                "model": "ac",
                "size": model.size,
                "depth": model.depth
            }, latest_ac_path)
        
                
        print(
            f"\rEpisode: {episode + 1}, Reward: {total_rewards:.2f}, "
            f"Avg100: {np.mean(totals[-100:]):.2f}, EW: {current_entropy_weight:.6f}",
            end=""
        )  

    model.eval()
    return totals



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
            checkpoint = torch.load(args.checkpoint, weights_only=False)
            model_type = checkpoint["model"]
            size = checkpoint["size"]
            depth = checkpoint["depth"]
        except Exception as e:
            model_type = "naive"
            print("Falling back to naive, could not load checkpoint")
    else:
        model_type = "naive"
        print("Falling back to naive, could not load checkpoint")


        
        

    env = gym.make("LunarLander-v2", render_mode="human")
    
    if model_type == "reinforce":
        model = PolicyNetwork(size, depth)
    elif model_type == "ac":
        model = ActorCritic(size, depth)

    if model_type != "naive":
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

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
            obs, reward, done, truncated, info = env.step(action)
            totals.append(reward)
            if done or truncated:
                break
        mean = sum(totals)
        print(f"Episode rewards total: {mean}")
        final_totals.append(mean)

    windows = np.max([100, episode])
    print("Final convolve: ", np.convolve(final_totals, np.ones(window), 'valid') / window)
    print("Final: ")
    print(np.mean(final_totals),np.std(final_totals),min(final_totals),max(final_totals))

    
    env.close()
