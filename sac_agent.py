import torch
import torch.nn.functional as F
from torch.optim import Adam
from model import GaussianPolicy, QNetwork
import utils

class soft_actor_critic_agent:
    def __init__(self, num_inputs, action_space,\
        device, hidden_size, seed, lr, gamma, tau, alpha):

        self.gamma = gamma
        self.tau
        self.alpha = alpha

        self.device = device
        self.seed = torch.manual_seed(seed)

        # It’s safe to call this function even if CUDA is not available; in that case, it is silently ignored.
        torch.cuda.manual_seed(seed)

        self.critic = QNetwork(seed, num_inputs, action_space.shape[0], hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        self.critic_target = QNetwork(seed, num_inputs, action_space.shape[0], hidden_size).to(self.device)
        utils.hard_update(self.critic_target, self.critic)