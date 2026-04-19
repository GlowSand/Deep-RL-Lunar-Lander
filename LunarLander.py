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

def get_ac_path(discount_factor, critic_weight, size, depth, lr, entropy_weight_start, entropy_weight_end, entropy_anneal_episodes):
    return Path(
        f"ac_df{discount_factor}"
        f"_cw{critic_weight:0.3f}"
        f"_size{size}"
        f"_depth{depth}"
        f"_lr{lr:0.6f}"
        f"_ews{entropy_weight_start:0.6f}"
        f"_ewe{entropy_weight_end:0.6f}"
        f"_ewa{entropy_anneal_episodes}"
    )

def train_actor_critic(model, optimizer, criterion, env, n_episodes=400, discount_factor=0.95, critic_weight=0.3, entropy_weight_start=0.001, entropy_weight_end=0.0001, entropy_anneal_episodes=400, resume=True):
    totals = []
    best_avg = -float("inf")

    ac_dir_path = get_ac_path(discount_factor, critic_weight, model.size, model.depth, optimizer.param_groups[0]['lr'], entropy_weight_start, entropy_weight_end, entropy_anneal_episodes)
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
        total_rewards = run_episode_and_train_ac(model, optimizer, criterion, env, discount_factor, critic_weight, current_entropy_weight, seed=seed)
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



import random


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
):
    return Path(
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
    save_replay_buffer=False,
    ac_model=None,
    ac_guidance_start=0.5,
    ac_guidance_end=0.0,
    ac_guidance_anneal_episodes=150,
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
    )
    dqn_dir_path.mkdir(exist_ok=True)

    latest_dqn_path = dqn_dir_path / Path("latest_dqn.pt")
    best_dqn_path = dqn_dir_path / Path("best_dqn.pt")

    start_episode = 0

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

        if save_replay_buffer and "replay_buffer" in checkpoint:
            replay_buffer = deque(checkpoint["replay_buffer"], maxlen=buffer_size)
    else:
        target_model.load_state_dict(policy_model.state_dict())

    if ac_model is not None:
        ac_model.eval()

    policy_model.train()
    target_model.eval()

    for episode in range(start_episode, n_episodes):
        seed = torch.randint(0, 2**32, size=()).item()

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
            seed=seed,
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
        }

        if save_replay_buffer:
            checkpoint_data["replay_buffer"] = list(replay_buffer)

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
        

    checkpoint = torch.load(path, weights_only=False)
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
    
    window = np.max([100, episode])
    print("Final convolve: ", np.convolve(final_totals, np.ones(window), 'valid') / window)
    print("Final: ")
    print(np.mean(final_totals),np.std(final_totals),min(final_totals),max(final_totals))

    
    env.close()
