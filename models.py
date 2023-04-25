import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 and classname.find('Group') == -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModel(nn.Module):
    def __init__(self, obs_space, action_space, memory_dim=64):
        super().__init__()
        
        self.obs_space = obs_space
        self.action_space = action_space
        self.memory_dim = memory_dim
        self.input_obs_shape = self.obs_space[0].shape
        
        self.layers = nn.Sequential(
            nn.Linear(self.input_obs_shape, 64),
            nn.ReLU(),
            nn.Linear(64, self.memory_dim)
        )

        # Define memory
        self.memory_rnn = nn.LSTMCell(self.memory_dim, self.memory_dim)

        self.embedding_size = self.semi_memory_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    def reset_hiddens(self):
        pass

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.memory_dim

    def forward(self, obs, memory):
        
        x = self.layers(obs) # model output

        hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])

        hidden = self.memory_rnn(x, hidden)

        embedding = hidden[0]
        memory = torch.cat(hidden, dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=-1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return {'dist': dist, 'value': value, 'memory': memory}

