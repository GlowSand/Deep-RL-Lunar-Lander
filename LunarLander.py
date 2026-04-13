import gymnasium as gym
import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(8, 48), nn.ReLU(), nn.Linear(48,48), nn.ReLU())

        self.actor_head = nn.Linear(48, 4)
        self.critic_head = nn.Linear(48, 1)

    def forward(self, state):
        features = self.body(state)
        return self.actor_head(features), self.critic_head(features).squeeze(-1)
    
def choose_action_and_evaluate(model, obs):
    state = torch.as_tensor(obs)
    logit, state_value = model(state)
    dist = torch.distributions.Categorical(logits=logit)
    entropy = dist.entropy()
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return int(action.item()), log_prob, state_value, entropy

env = gym.make("LunarLander-v2", render_mode="human")


ac_model = ActorCritic()
ac_model.load_state_dict(torch.load("best_actor_critic.pth"))

for episode in range(200):
    total_rewards = 0
    obs,info = env.reset()
    while True:
        action = choose_action_and_evaluate(ac_model, obs)
        obs, reward, done, truncated, info = env.step(action[0])
        total_rewards += reward
        if done or truncated:
            break
