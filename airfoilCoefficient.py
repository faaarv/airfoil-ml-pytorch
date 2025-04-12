import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
import time
import matplotlib.pyplot as plt

# Load the data
profiles = np.load("profiles.npy")
conditions = np.load("conditions.npy")
coefficients = np.load("coefficients.npy")
pressures = np.load("pressures.npy")

# Create features and labels
features = np.empty((profiles.shape[0], profiles.shape[1] + conditions.shape[1]))
labels = np.empty((coefficients.shape[0], coefficients.shape[1] + pressures.shape[1]))

features[:, :200] = profiles
features[:, 200] = np.log10(conditions[:, 0])
features[:, 201] = conditions[:, 1]
features[:, 202] = conditions[:, 2]

labels[:, :4] = coefficients
labels[:, 4:] = pressures

# Split data
X_train_val, X_test, y_train_val, y_test = train_test_split(features, labels, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15 / 0.85, random_state=42)

# PreProcess
profiles_scaler = StandardScaler()
Re_scaler = MinMaxScaler()
AoA_scaler = StandardScaler()
Ncrit_scaler = MinMaxScaler()

X_train[:, :200] = profiles_scaler.fit_transform(X_train[:, :200])
X_val[:, :200] = profiles_scaler.transform(X_val[:, :200])
X_test[:, :200] = profiles_scaler.transform(X_test[:, :200])

X_train[:, 200] = Re_scaler.fit_transform(X_train[:, 200].reshape(-1, 1)).flatten()
X_val[:, 200] = Re_scaler.transform(X_val[:, 200].reshape(-1, 1)).flatten()
X_test[:, 200] = Re_scaler.transform(X_test[:, 200].reshape(-1, 1)).flatten()

X_train[:, 201] = AoA_scaler.fit_transform(X_train[:, 201].reshape(-1, 1)).flatten()
X_val[:, 201] = AoA_scaler.transform(X_val[:, 201].reshape(-1, 1)).flatten()
X_test[:, 201] = AoA_scaler.transform(X_test[:, 201].reshape(-1, 1)).flatten()

X_train[:, 202] = Ncrit_scaler.fit_transform(X_train[:, 202].reshape(-1, 1)).flatten()
X_val[:, 202] = Ncrit_scaler.transform(X_val[:, 202].reshape(-1, 1)).flatten()
X_test[:, 202] = Ncrit_scaler.transform(X_test[:, 202].reshape(-1, 1)).flatten()

CLcoefficients_scaler = MinMaxScaler()
Cdcoefficients_scaler = MinMaxScaler()
Cmcoefficients_scaler = MinMaxScaler()
pressures_scaler = MinMaxScaler()



y_train[:, 0] = CLcoefficients_scaler.fit_transform(y_train[:, 0].reshape(-1, 1)).flatten()
y_val[:, 0] = CLcoefficients_scaler.transform(y_val[:, 0].reshape(-1, 1)).flatten()
y_test[:, 0] = CLcoefficients_scaler.transform(y_test[:, 0].reshape(-1, 1)).flatten()

y_train[:, 1:3] = Cdcoefficients_scaler.fit_transform(y_train[:, 1:3])
y_val[:, 1:3] = Cdcoefficients_scaler.transform(y_val[:, 1:3])
y_test[:, 1:3] = Cdcoefficients_scaler.transform(y_test[:, 1:3])

y_train[:, 3] = Cmcoefficients_scaler.fit_transform(y_train[:, 3].reshape(-1, 1)).flatten()
y_val[:, 3] = Cmcoefficients_scaler.transform(y_val[:, 3].reshape(-1, 1)).flatten()
y_test[:, 3] = Cmcoefficients_scaler.transform(y_test[:, 3].reshape(-1, 1)).flatten()

y_train[:, 4:] = pressures_scaler.fit_transform(np.clip(y_train[:, 4:], -2, 2))
y_val[:, 4:] = pressures_scaler.transform(np.clip(y_val[:, 4:], -2, 2))
y_test[:, 4:] = pressures_scaler.transform(np.clip(y_test[:, 4:], -2, 2))

# PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

batch_size = 256
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)


# PyTorch Model
class DynamicAerodynamicDNN(nn.Module):
    def __init__(self, input_dim, hidden_units_per_layer, output_units, activation):
        super(DynamicAerodynamicDNN, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_units_per_layer[0]))
        layers.append(self.get_activation(activation))

        # Hidden layers
        for i in range(1, len(hidden_units_per_layer)):
            layers.append(nn.Linear(hidden_units_per_layer[i - 1], hidden_units_per_layer[i]))
            layers.append(self.get_activation(activation))

        # Output layer
        layers.append(nn.Linear(hidden_units_per_layer[-1], output_units))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def get_activation(self, activation_name):
        if activation_name == 'relu':
            return nn.ReLU()
        elif activation_name == 'tanh':
            return nn.Tanh()
        elif activation_name == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation_name}")

# Define model settings
model_settings = {
    'hidden_units_per_layer': [128, 128, 128],  
    'activation': 'relu',
    'output_units': labels.shape[1],
    'learning_rate': 0.001,
    'epochs': 200,
    'patience': 10  # Early stopping patience
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DynamicAerodynamicDNN(
    input_dim=X_train.shape[1],
    hidden_units_per_layer=model_settings['hidden_units_per_layer'],
    output_units=model_settings['output_units'],
    activation=model_settings['activation']
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=model_settings['learning_rate'])

best_val_loss = float('inf')
patience_counter = 0
train_losses = []
val_losses = []

for epoch in range(model_settings['epochs']):
    model.train()
    batch_train_losses = []
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        batch_train_losses.append(loss.item())
    train_loss = np.mean(batch_train_losses)
    train_losses.append(train_loss)

    model.eval()
    batch_val_losses = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            batch_val_losses.append(loss.item())
    val_loss = np.mean(batch_val_losses)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{model_settings['epochs']}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= model_settings['patience']:
            print("Early stopping triggered.")
            break

# Plot training and validation losses
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid()
plt.show()
