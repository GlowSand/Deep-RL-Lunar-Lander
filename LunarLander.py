import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

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
    state = torch.as_tensor(obs)
    logit = model(state)
    dist = torch.distributions.Categorical(logits=logit)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return int(action.item()), log_prob

def choose_action_and_evaluate(model, obs): #Action with eval for AC
    state = torch.as_tensor(obs)
    logit, state_value = model(state)
    dist = torch.distributions.Categorical(logits=logit)
    entropy = dist.entropy()
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return int(action.item()), log_prob, state_value, entropy


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

def train_reinforce(model, optimizer, env, n_episodes, discount_factor,  resume=True):
    model.train()
    reinforce_dir_path = Path(f"reinforce_df{discount_factor}_size{model.size}_depth{model.depth}_lr{optimizer.param_groups[0]['lr']:0.6f}")

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

def get_target_value(model, next_obs, reward, done, truncated, discount_factor):
    with torch.inference_mode():
        _, _, next_state_value, _ = choose_action_and_evaluate(model, next_obs)

    running = 0.0 if (done or truncated) else 1.0
    target_value = reward + running * discount_factor * next_state_value
    return target_value

def run_episode_and_train_ac(model, optimizer, criterion, env, discount_factor, critic_weight, entropy_weight, seed=None):
    obs, _info = env.reset(seed=seed)
    total_rewards = 0
    while True:
        action, log_prob, state_value, entropy = choose_action_and_evaluate(model, obs)
        next_obs, reward, done, truncated, _info = env.step(action)
        target_value = get_target_value(model, next_obs, reward, done, truncated, discount_factor)
        ac_training_step(model, optimizer, criterion, state_value, target_value, log_prob, entropy, critic_weight, entropy_weight)
        total_rewards += reward
        if  done or truncated:
            return total_rewards
        obs = next_obs

def train_actor_critic(model, optimizer, criterion, env, n_episodes=400, discount_factor=0.95, critic_weight=0.3, entropy_weight=0.0005, resume=True):
    totals = []
    best_avg = -float("inf")

    ac_dir_path = Path(f"ac_df{discount_factor}_cw{critic_weight:0.3f}_size{model.size}_depth{model.depth}_lr{optimizer.param_groups[0]['lr']:0.6f}_ew{entropy_weight:0.6}")

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
        total_rewards = run_episode_and_train_ac(model, optimizer, criterion, env, discount_factor, critic_weight, entropy_weight,  seed=seed)
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
        
        
                
        print(f"\rEpisode: {episode + 1}, Rewards: {total_rewards}", end=" ")
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

        
        

    env = gym.make("LunarLander-v2", render_mode="human")
    
    if model_type == "reinforce":
        model = PolicyNetwork(size, depth)
    elif model_type == "ac":
        model = ActorCritic(size, depth)

    if model_type != "naive":
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
    
    for episode in range(args.episodes):
        episode_seed = None if args.seed is None else args.seed + episode
        obs,info = env.reset(seed=episode_seed)
        while True:
            if (model_type == "naive"):
                action = basic_policy(obs)
            elif (model_type == "reinforce"):
                action = choose_action(model, obs)[0]
            elif (model_type == "ac"):
                action = choose_action_and_evaluate(model, obs)[0]
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                break
    env.close()
