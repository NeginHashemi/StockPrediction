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
    def __init__(self, obs_space, action_space, memory_dim=64, arch='linear', c=100, w=14):
        super().__init__()
        
        self.obs_space = obs_space
        self.action_space = action_space
        self.memory_dim = memory_dim
        self.input_obs_shape = self.obs_space[0].shape[0]
        self.arch = arch
        self.window_size = w
        self.past_window_size = c

        if self.arch == 'linear':
        
            self.layers = nn.Sequential(
                nn.Linear(self.input_obs_shape, 64),
                nn.ReLU(),
                nn.Linear(64, self.memory_dim)
            )

        else:
            num_features = int(self.input_obs_shape//self.past_window_size)
            self.conv = nn.Conv2d(kernel_size=(1, num_features), out_channels=1, 
            in_channels=1, stride=(1, num_features))
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(self.past_window_size, 4),
                2
            )

            self.fc1 = nn.Linear(self.past_window_size, self.memory_dim)

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

        bs = obs.shape[0]

        if self.arch == 'linear':
            x = self.layers(obs) # model output
        elif self.arch == 'attention':
            embedded = self.conv(obs.reshape(bs, 1, 1, -1))
            output = self.transformer(embedded.squeeze(2))
            x = self.fc1(output.squeeze(1))

        hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])

        hidden = self.memory_rnn(x, hidden)

        embedding = hidden[0]
        memory = torch.cat(hidden, dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=-1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return {'dist': dist, 'value': value, 'memory': memory}


class RModel(nn.Module):
    def __init__(self, in_dim=9, memory_dim=64):
        super().__init__()
        
        self.in_dim = in_dim
        self.memory_dim = memory_dim
        
        self.layers = nn.Sequential(
            nn.Linear(self.in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.memory_dim)
        )

        # Define memory
        self.memory_rnn = nn.LSTM(self.memory_dim, self.memory_dim, 1, batch_first=True)

        self.final = nn.Linear(self.memory_dim, 1)
        
        # Initialize parameters correctly
        self.apply(initialize_parameters)

    def forward(self, x):
        
        bs, L, d = x.shape
        x = self.layers(x.view(bs*L, d)) # model output

        output, _ = self.memory_rnn(x.view(bs, L, self.memory_dim))

        output = self.final(output[:, -1, :]).reshape([bs,])

        return output