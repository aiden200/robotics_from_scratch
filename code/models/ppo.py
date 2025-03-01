import torch
import torch.nn as nn
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import os


class PPOConfig:
    def __init__(
            self,
            input_shape: tuple,
            epochs: int = 15,
            batch: int = 4096,
            horizon: int = 512,
            loss_type: str = "clipped",
            discount_value: float = .99,
            num_actors: int = 32,
            epsilon: float = 0.1,
            hidden_dim: int = 1024,
            action_space: str = "discrete",
            action_dim: int = 4,
            value_func_dim: int = 2048,
            raw_pixels: bool = False,
            kl_dtarg: Optional[float] = .01,
            beta: Optional[float] = .3,
            gae_parameter: Optional[float] = .95,
            c1: Optional[float] = 1.0,
            c2: Optional[float] = .01
            ):
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch = batch
        self.horizon = horizon # T, timestamps we look ahead
        self.loss_type = loss_type # clipped, normal, kl_penalized
        self.discount_value = discount_value
        self.num_actors = num_actors
        self.epsilon = epsilon #Clipping coefficient
        self.kl_dtarg = kl_dtarg # KL divergance target ratio
        self.beta = beta # How much KL divergence affects the penalty
        self.gae_parameter = gae_parameter # lambda
        self.c1 = c1 # Value function coefficient
        self.c2 = c2 # Entropy coefficient
        self.device = "cuda" if torch.cuda.is_available else "cpu"
        self.hidden_dim = hidden_dim
        self.action_space = action_space # discrete or continuous
        self.action_dim = action_dim
        self.raw_pixels = raw_pixels # If we input an image or not
        self.value_function_dim = value_func_dim


class PPOActor(nn.Module):
    def __init__(self, config: PPOConfig):
        super().__init__()

        self.actors = config.num_actors
        self.gae_parameter = config.gae_parameter
        self.discount_value = config.discount_value
        self.horizon = config.horizon
        self.input_shape = config.input_shape
        self.device = config.device
        self.hidden_dim = config.hidden_dim
        self.raw_pixels = config.raw_pixels
        self.action_space = config.action_space

        if self.raw_pixels:
            # Project the 2d image to a hidden state
            self.input_encode = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3,stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten()
            )

            self.flattened_size = self._get_conv_output(self.input_shape)
            self.fc = nn.Linear(self.flattened_size, self.hidden_dim)
        else:
            self.fc = nn.Linear(self.input_shape, self.hidden_dim)


        if config.action_space == "discrete":
            self.action_encode = nn.Linear(self.hidden_dim, config.action_dim)

        elif config.action_space == "continuous":

            self.mu_head = nn.Linear(self.hidden_dim, config.action_dim)  
            self.log_std = nn.Parameter(torch.zeros(config.action_dim))
        else:
            raise ValueError(f"Action space: {config.action_space} not implemented")
        

    def _get_conv_output(self, shape):
        """Computes the output size of the CNN dynamically."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)  # Batch size 1
            conv_out = self.input_encode(dummy_input)

            # C * W * H -> 1: gets rid of the batch
            return int(torch.prod(torch.tensor(conv_out.shape[1:]))) # Flattened size


    def forward(self, x):

        if self.raw_pixels:
            assert x.dim() == 4, f"Expected input shape (B, C, H, W), got {x.shape}"
            x = self.input_encode(x)
        
        if not self.raw_pixels and x.dim() == 1:
            x = x.unsqueeze(0) # adding batch dimension 
        
        hidden_states = self.fc(x)

        if self.action_space == "discrete":
            logits = self.action_encode(hidden_states)
            out = torch.softmax(logits, dim=-1)
            return out, None, None
        elif self.action_space == "continuous":
            # For trying to explore, lower std is more deterministic, higher more exploration
            # Mu represents where the highest probability of the action lies
            mu = self.mu_head(hidden_states)
            std = torch.exp(self.log_std)
            distribution = torch.distributions.Normal(mu, std)
            entropy = distribution.entropy().sum(dim=-1)
            action = distribution.sample()
            log_prob = distribution.log_prob(action).sum(dim=-1) # sum over action dimensions
            return action, log_prob, entropy


class PPOCritic(nn.Module):
    def __init__(self, config: PPOConfig):
        super().__init__()
        # Take in the state information, and output a single value
        # We use this to estimate the value of the state.
        # In training, the Advantage function A_t will be calculated by A_t = R_s - V(s)
        self.input_shape = config.input_shape
        self.hidden_dim = config.value_function_dim
        self.raw_pixels = config.raw_pixels

        if self.raw_pixels:
            # Project the 2d image to a hidden state
            self.input_encode = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3,stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten()
            )

            self.flattened_size = self._get_conv_output(self.input_shape)
            self.fc = nn.Linear(self.flattened_size, self.hidden_dim)
        else:
            self.fc = nn.Linear(self.input_shape, self.hidden_dim)
        
        self.value_lin = nn.Linear(self.hidden_dim, 1)
    
    def _get_conv_output(self, input_shape):
        with torch.no_grad():
            dummy_tensor = torch.ones(1, *input_shape, dtype=torch.float32)
            x = self.input_encode(dummy_tensor)
            # (H, W, C)
            shape = x.shape[1:]
            # H * W * C
            shape = int(torch.prod(torch.tensor(shape)))
            return shape

    def forward(self, x):
        if self.raw_pixels:
            x = self.input_encode(x)
        if not self.raw_pixels and x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
        
        x = self.fc(x)
        out = self.value_lin(x)
        return out
    
    
        
    
class EnvironmentConnected:
    def __init__(self):
        pass
    
    def get_reward(self, state):
        return 1
    
    def reset(self):
        pass
    
    def step(self, action):
        pass
        # return new_state, reward, done
    


class PPOClippedLoss(nn.Module):
    def __init__(self, config : PPOConfig):
        super().__init__()
        self.epsilon = config.epsilon
    
    def compute_loss(
        self,
        old_log_probs,
        new_log_probs,
        advantages
        ):
        # log(n_probs) - log(o_probs) = log(n_probs/o_probs)
        log_r = new_log_probs - old_log_probs
        # n_probs/o_probs = exp^(log(n_probs/o_probs))
        r = torch.exp(log_r)
        r_a = r * advantages
        clipped = torch.clamp(r, 1 - self.epsilon, 1 + self.epsilon) * advantages
        losses = torch.min(r_a, clipped)
        return -losses.mean()

class PPOLossNoPenalize(nn.Module):
    def __init__(self):
        super().__init__()
    
    def compute_loss(
        self,
        old_log_probs,
        new_log_probs,
        advantages
    ):
        
        r = torch.exp(new_log_probs - old_log_probs)
        losses = r * advantages
        return -losses.mean()

class PPOLossKL(nn.Module):
    def __init__(self, config: PPOConfig):
        super().__init__()
        self.beta = config.beta
        self.dtarg = config.kl_dtarg
    
    def compute_loss(
        self,
        old_log_probs,
        new_log_probs,
        advantages
    ):
        r_a = torch.exp(new_log_probs - old_log_probs) * advantages
        d_kl = (old_log_probs - new_log_probs).mean()
        
        loss = -r_a.mean() + self.beta * d_kl
        
        if d_kl < (self.dtarg/1.5):
            self.beta = max(self.beta / 2, 1e-5)
        elif d_kl > (self.dtarg*1.5):
            self.beta = min(self.beta * 2, 10) 
        
        return loss
        
        



class PPO(nn.Module):
    def __init__(self, input_shape, env):
        super().__init__()
        self.config = PPOConfig(input_shape=input_shape)
        self.epochs = self.config.epochs
        self.actor = PPOActor(self.config).to(self.config.device)
        self.critic = PPOCritic(self.config).to(self.config.device)
        self.horizon = self.config.horizon
        self.gae_value = self.config.gae_parameter
        self.discount = self.config.discount_value
        self.env = env
        self.c1 = self.config.c1
        self.c2 = self.config.c2
        
        if self.config.loss_type == "clipped":
            self.loss_surrogate = PPOClippedLoss(self.config, self.critic)
        elif self.config.loss_type == "normal":
            self.loss_surrogate = PPOLossNoPenalize(self.config)
        elif self.config.loss_type == "kl_penalized":
            self.loss_surrogate = PPOLossKL(self.config)
        else:
            raise ValueError(f"Loss {self.config.loss_type} not implemented")

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
    
    def train(self):
        old_log_probs = None
        
        for epoch in range(self.epochs):
            states, entropies, log_probs, rewards, values, finished_states = self.collect_trajectories()
            
            # Store old log probs before updating policy
            # .detach() removes the gradient tracking, .clone() avoids in place changes
            if old_log_probs is None:
                old_log_probs = log_probs.clone().detach()
                
            advantages = self.calculate_advantages(rewards, values)
            
            self.update_policy(entropies, old_log_probs, log_probs, advantages, rewards, values, finished_states)
            
            old_log_probs = log_probs.clone().detach()
    
    def collect_trajectories(self):
        # Environment interactions should remain on CPU, Actor Critic functions in GPU
        states = [self.env.reset() for _ in range(self.config.num_actors)]
        entropies, log_probs, rewards, values, finished_states = [], [], [], [], []
        
        for t in range(self.horizon):
            state_tensors = torch.tensor(states, dtype=torch.float32).to(self.config.device)
            
            # sample action from distribution
            with torch.no_grad():
                actions, log_probs, entropies = self.actor(state_tensors)
                # get value for current state
                values = self.critic(state_tensors)
            
            
            # Environment needs to support multiple action spaces. TODO if not supported
            next_states, rewards, dones = self.env.step(actions.cpu().numpy()) # If env is on cpu
            
            
            
            
            states.append(state)
            entropies.append(entropy)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            finished_states.append((done))
            
            state = next_state
            
            # In finish state
            if done:
                break
        
        
        return (
            torch.tensor(states, dtype=torch.float32).to(self.config.device),
            torch.tensor(entropies, dtype=torch.float32).to(self.config.device),
            torch.tensor(log_probs, dtype=torch.float32).to(self.config.device),
            torch.tensor(rewards, dtype=torch.float32).to(self.config.device),
            torch.tensor(values, dtype=torch.float32).to(self.config.device),
            torch.tensor(finished_states, dtype=torch.float32).to(self.config.device),
        )
    
    def calculate_advantages(self, rewards, values, finished_states):
        '''
        rewards: rewards for each time step t
        values: the value function return for each time step t
        finished_states: if finished_states[t] == 1, it means we are in a finished state, and our reward should be 0
        '''
        
        advantages = torch.zeros_like(rewards, dtype=torch.float32, device=self.config.device)
        advantage = 0
        
        #GAE values rely on future rewards, so we iterate backwards, starting from time T, all the way back to time 0
        for t in range(self.horizon-2, -1, -1):
            curr_advantage = rewards[t] + self.discount * values[t+1] * (1-finished_states[t]) - values[t]
            advantage = curr_advantage + (1 - finished_states[t]) * self.discount * self.gae_value * advantage
            advantages[t] = advantage
            
        return advantages
    
    def update_policy(self, entropies, old_log_probs, log_probs, advantages, rewards, values):
        
        surrogate_loss = self.loss_surrogate.compute_loss(old_log_probs, log_probs, advantages)
        value_loss = nn.functional.mse_loss(rewards, values)
        entropy_loss = -torch.mean(entropies)
        
        loss = surrogate_loss - self.c1 * value_loss + -self.c2 * entropy_loss
        
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()


class PPOTester:
    def __init__(self):
        pass
    
    def test_actor(self):
        sensor_input_size = 10 
        image_input_shape = (3, 64, 64)  # (Channels, Height, Width) for CNN input



        # Test with discrete action space and vector input
        config = PPOConfig(action_space="discrete", action_dim=5, raw_pixels=False)
        actor = PPOActor(config, input_size=sensor_input_size).to(config.device)
        # Dummy sensor data (batch_size=2, features=10)
        sensor_data = torch.randn(2, sensor_input_size).to(config.device)
        # Run forward pass
        action_probs, _, _ = actor(sensor_data)
        # Assertions
        assert action_probs.shape == (2, config.action_dim), f"Expected shape (2, {config.action_dim}), got {action_probs.shape}"
        assert torch.all(action_probs >= 0) and torch.all(action_probs <= 1), "Probabilities must be between 0 and 1"
        print("✅ Discrete MLP Test Passed!")


        config = PPOConfig(action_space="continuous", action_dim=3, raw_pixels=False)
        actor = PPOActor(config, input_size=sensor_input_size).to(config.device)
        # Dummy sensor data (batch_size=2, features=10)
        sensor_data = torch.randn(2, sensor_input_size).to(config.device)
        # Run forward pass
        action, log_prob, dist = actor(sensor_data)
        # Assertions
        assert action.shape == (2, config.action_dim), f"Expected shape (2, {config.action_dim}), got {action.shape}"
        assert log_prob.shape == (2,), f"Expected log_prob shape (2,), got {log_prob.shape}"
        assert torch.isfinite(action).all(), "Actions should not contain NaN or Inf"
        print("✅ Continuous MLP Test Passed!")


        # Test with discrete action space and CNN input
        config = PPOConfig(action_space="discrete", action_dim=6, raw_pixels=True)
        actor = PPOActor(config, input_size=image_input_shape).to(config.device)
        # Dummy image data (batch_size=2, channels=3, height=64, width=64)
        image_data = torch.randn(2, *image_input_shape).to(config.device)
        # Run forward pass
        action_probs, _, _ = actor(image_data)
        # Assertions
        assert action_probs.shape == (2, config.action_dim), f"Expected shape (2, {config.action_dim}), got {action_probs.shape}"
        assert torch.all(action_probs >= 0) and torch.all(action_probs <= 1), "Probabilities must be between 0 and 1"
        print("✅ Discrete CNN Test Passed!")


        # Test with continuous action space and CNN input
        config = PPOConfig(action_space="continuous", action_dim=2, raw_pixels=True)
        actor = PPOActor(config, input_size=image_input_shape).to(config.device)
        # Dummy image data (batch_size=2, channels=3, height=64, width=64)
        image_data = torch.randn(2, *image_input_shape).to(config.device)
        # Run forward pass
        action, log_prob, dist = actor(image_data)
        # Assertions
        assert action.shape == (2, config.action_dim), f"Expected shape (2, {config.action_dim}), got {action.shape}"
        assert log_prob.shape == (2,), f"Expected log_prob shape (2,), got {log_prob.shape}"
        assert torch.isfinite(action).all(), "Actions should not contain NaN or Inf"
        print("✅ Continuous CNN Test Passed!")

if __name__ == "__main__":
    tester = PPOTester()
    tester.test_actor()
