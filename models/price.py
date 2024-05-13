import torch
import torch.nn as nn
import numpy as np

# Get prices from price_dict.json
import json

with open("price_dict.json", "r") as file:
    price_dict = json.load(file)
    
prices = list(price_dict.values())

# Scale prices
# prices = np.array(prices)
# prices = (prices - prices.mean()) / prices.std()

# Function to create sequences
def create_sequences(data, N):
    X, y = [], []
    for i in range(len(data) - N):
        X.append(data[i:i+N])
        y.append(data[i+N])
    return np.array(X), np.array(y)

# Create input and target sequences
N = 10  # Using the last 3 prices to predict the next one
X, y = create_sequences(prices, N)

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # Add feature dimension
y = torch.tensor(y, dtype=torch.float32)

import torch.nn.init as init

class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers):
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=input_dim,
            nhead=num_heads,
            num_decoder_layers=num_layers,
            num_encoder_layers=num_layers,
            batch_first=False
        )
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, src):
        src = src.permute(1, 0, 2)  # Transformer expects (S, N, E)
        output = self.transformer(src, src)
        return self.linear(output[-1, :, :])
    
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.lstm.weight_ih_l0)
        init.orthogonal_(self.lstm.weight_hh_l0)
        init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0.01)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.linear(output[:, -1, :])
    
    
    
    
    
    
    
    
    
    
    
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


def train_model(model, X, y, loss_function, optimizer, epochs):
    for i in range(epochs):
        for seq, labels in zip(X, y):
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if i%25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
            


    
    
    
    
    
    
    
    

# Model parameters
input_dim = 1  # Since we are just using price
num_heads = 1
num_layers = 100
# dropout = 0.1

# Instantiate the model
# model = TransformerModel(input_dim, num_heads, num_layers)
# model = LSTMModel(input_dim, num_layers)
model = LSTMModel(input_dim, num_layers, 1)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Custom LR scheduler
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)


# Train the model
num_epochs = 50


train_model(model, X, y, criterion, optimizer, num_epochs)

# Run model over the X data and save the output
predictions = []
for seq in X:
    with torch.no_grad():
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        predictions.append(model(seq).item())

    
# print(output)




# for epoch in range(num_epochs):
#     model.train()
#     optimizer.zero_grad()
#     output = model(X)
#     loss = criterion(output.squeeze(-1), y)  # Remove unnecessary dimensions
#     loss.backward()
#     optimizer.step()
#     # scheduler.step()

#     if (epoch + 1) % 100 == 0:
#         print(f"Epoch {epoch + 1}, Loss: {loss.item()}, Output: {output[-1].item()}, Actual: {y[-1].item()}")


    
# Plot the actuals and outpout over time
import matplotlib.pyplot as plt

# output = model(X).detach().numpy().squeeze(-1)

# print(output)

plt.plot(prices, label="Actual")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.show()


print(f"X: {X[0]}, y: {y[0]}")
print('********************')
print(f"X: {X[1]}, y: {y[1]}")
print('********************')
print(f"X: {X[2]}, y: {y[2]}")
print('********************')
