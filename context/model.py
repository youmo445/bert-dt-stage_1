import torch
import torch.nn as nn
import torch.nn.functional as F


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
    
class RNNContextEncoder(nn.Module):
    def __init__(self, state_dim, action_dim, context_dim, context_hidden_dim):
        super(RNNContextEncoder, self).__init__()

        self.state_encoder = nn.Sequential(nn.Linear(state_dim, context_dim), nn.ReLU())
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, context_dim), nn.ReLU())
        self.reward_encoder = nn.Sequential(nn.Linear(1, context_dim), nn.ReLU())

        self.gru = nn.GRU(input_size=3*context_dim, hidden_size=context_hidden_dim, num_layers=1)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        # output layer, output z
        self.context_output = nn.Linear(context_hidden_dim, context_dim)

        self.apply(weights_init_)

    def forward(self, states, actions, rewards):
        """
        Actions, states, rewards should be given in form [sequence_len * batch_size * dim].
        """
        # shape should be: sequence_len x batch_size x dim
        states = states.reshape((-1, *states.shape[-2:]))#(200,100,dim)
        actions = actions.reshape((-1, *actions.shape[-2:]))
        rewards = rewards.reshape((-1, *rewards.shape[-2:]))

        # extract features for states, actions, rewards
        hs = self.state_encoder(states)
        ha = self.action_encoder(actions)
        hr = self.reward_encoder(rewards)
        h = torch.cat((ha, hs, hr), dim=-1)

        # gru_output: [seq_len * batch_size * hidden_dim]
        gru_output, _ = self.gru(h)#(200, 100, hidden_dim)
        #(100, hidden_dim)
        contexts = self.context_output(gru_output[-1])
        return contexts


class RewardDecoder(nn.Module):
    def __init__(self, state_dim, action_dim, context_dim, context_hidden_dim):
        super(RewardDecoder, self).__init__()

        self.state_encoder = nn.Sequential(nn.Linear(state_dim, context_dim), nn.ReLU())
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, context_dim), nn.ReLU())
        
        self.linear1 = nn.Linear(context_dim*3, context_hidden_dim)
        self.linear2 = nn.Linear(context_hidden_dim, context_hidden_dim)
        self.linear3 = nn.Linear(context_hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action,  context):
        # extract features for states, actions
        hs = self.state_encoder(state)
        ha = self.action_encoder(action)
        h = torch.cat((hs, ha, context), dim=-1)

        h = F.relu(self.linear1(h))
        h = F.relu(self.linear2(h))
        reward_predict = self.linear3(h)
        
        return reward_predict
class StateDecoder(nn.Module):
    def __init__(self, state_dim, action_dim, context_dim, context_hidden_dim):
        super(StateDecoder, self).__init__()

        self.state_encoder = nn.Sequential(nn.Linear(state_dim, context_dim), nn.ReLU())
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, context_dim), nn.ReLU())
        self.reward_encoder = nn.Sequential(nn.Linear(1, context_dim), nn.ReLU())
        
        self.linear1 = nn.Linear(context_dim*3, context_hidden_dim)
        self.linear2 = nn.Linear(context_hidden_dim, context_hidden_dim)
        self.linear3 = nn.Linear(context_hidden_dim, state_dim)

        self.apply(weights_init_)

    def forward(self, state, action,  context):
        # extract features for states, actions
        hs = self.state_encoder(state)
        ha = self.action_encoder(action)
        h = torch.cat((hs, ha,context), dim=-1)

        h = F.relu(self.linear1(h))
        h = F.relu(self.linear2(h))
        state_predict = self.linear3(h)
        
        return state_predict

   
class descriptionsEecoder(nn.Module):
    def __init__(self, bert_dim, hidden_dim):
        super(descriptionsEecoder, self).__init__()
        
        self.linear1 = nn.Linear(bert_dim, 256)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(256, hidden_dim)

        self.apply(weights_init_)

    def forward(self, desc):
        # extract features for states, actions
        x = self.relu(self.linear1(desc))
        x = self.linear2(x)
        
        return x







