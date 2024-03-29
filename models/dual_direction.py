import torch
import torch.nn as nn
import torch.optim as optim


class DualDirectionalCANetwork(nn.Module):
    def __init__(self, num_rules):
        super(DualDirectionalCANetwork, self).__init__()
        # Forward pathway
        self.forward_net = nn.Sequential(
            nn.Linear(3, 10),  # Example sizes
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
        # Backward pathway
        self.encoder = nn.LSTM(input_size=1, hidden_size=20, batch_first=True)
        self.decoder_state = nn.Linear(20, 1)
        self.decoder_rule = nn.Linear(20, num_rules)
    
    def forward(self, x, sequence=None):
        if sequence is None:
            # Forward prediction
            return self.forward_net(x)
        else:
            # Backward inference
            _, (hidden, _) = self.encoder(sequence)
            prev_state = self.decoder_state(hidden.squeeze(0))
            rule = self.decoder_rule(hidden.squeeze(0))
            return prev_state, rule

# Example usage
num_rules = 256  # For elementary CA, there are 256 possible rules
model = DualDirectionalCANetwork(num_rules=num_rules)

# Example forward input
forward_input = torch.tensor([[1, 0, 1]], dtype=torch.float)
forward_output = model(forward_input)

# Example backward input (sequence of states)
sequence_input = torch.rand((1, 10, 1))  # Example: batch_size=1, sequence_length=10
prev_state, rule = model(None, sequence_input)

# Define loss functions and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Example forward training
optimizer.zero_grad()
output = model(forward_input)
target = torch.rand(1, 1)  # Example target
loss = criterion(output, target)
loss.backward()
optimizer.step()
print(f"Forward Loss: {loss.item()}")

# Example backward training
optimizer.zero_grad()
prev_state, rule = model(None, sequence_input)
prev_state_target = torch.rand(1, 1)  # Example target
rule_target = torch.rand(1, num_rules)  # Example target
prev_state_loss = criterion(prev_state, prev_state_target)
rule_loss = criterion(rule, rule_target)
loss = prev_state_loss + rule_loss
loss.backward()
optimizer.step()
print(f"Backward Loss: {loss.item()}")

    