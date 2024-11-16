import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm  # Import tqdm for progress bars

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the optimized CNNAgent with fewer channels to reduce memory usage
class CNNAgent(nn.Module):
    def __init__(self):
        super(CNNAgent, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 16, 64)  # Adjusted to 32 * (64 // 4) * (64 // 4)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Dynamically infer the size after pooling
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the QMIX Network
class QMIXNetwork(nn.Module):
    def __init__(self, num_agents):
        super(QMIXNetwork, self).__init__()
        self.fc1 = nn.Linear(num_agents, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, q_values):
        x = F.relu(self.fc1(q_values))
        return self.fc2(x)

# Data loading with smaller batch size
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.ImageFolder(root=r"C:\Users\sriau\Downloads\PreddRNN\PredRNN\data", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Reduced batch size

# Initialize agents and QMIX
num_agents = 36  # One for each ASL symbol (0-9, a-z)
agents = [CNNAgent().to(device) for _ in range(num_agents)]
qmix = QMIXNetwork(num_agents=num_agents).to(device)

# Optimizers
agent_optimizers = [optim.Adam(agent.parameters(), lr=0.001) for agent in agents]
qmix_optimizer = optim.Adam(qmix.parameters(), lr=0.001)

# Training Loop
num_epochs = 5
for epoch in range(num_epochs):
    total_loss = 0
    # Add tqdm to the training loader for progress tracking
    epoch_iterator = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")

    for images, labels in epoch_iterator:
        images, labels = images.to(device), labels.to(device)

        # Collect Q-values from each agent with memory-efficient stacking
        q_values = []
        agent_losses = []
        for i, agent in enumerate(agents):
            agent_q_value = agent(images).squeeze(1)  # Shape: [batch_size]
            q_values.append(agent_q_value)

            # Compute target: 1 if the label matches the agent's symbol, else 0
            target = (labels == i).float().to(device)
            agent_loss = F.mse_loss(agent_q_value, target)
            agent_losses.append(agent_loss)

        # Stack Q-values without tracking gradients to save memory
        with torch.no_grad():
            q_values = torch.stack(q_values, dim=1).clone().detach()

        # Compute QMIX output
        qmix_output = qmix(q_values.to(device))  # Shape: [batch_size, 1]
        qmix_target = torch.ones_like(qmix_output)  # Target Q-value
        qmix_loss = F.mse_loss(qmix_output, qmix_target)

        # Sum all losses (agents and QMIX) and perform a single backward pass
        total_loss = qmix_loss + sum(agent_losses)

        # Zero gradients for all optimizers
        qmix_optimizer.zero_grad()
        for opt in agent_optimizers:
            opt.zero_grad()

        # Backward and optimize
        total_loss.backward()

        # Step all optimizers
        qmix_optimizer.step()
        for opt in agent_optimizers:
            opt.step()

        # Record total loss value
        total_loss_value = qmix_loss.item() + sum([loss.item() for loss in agent_losses])
        total_loss += total_loss_value

        # Update tqdm progress bar with current loss
        epoch_iterator.set_postfix(loss=total_loss_value / len(train_loader))

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")
# Switch models to evaluation mode
for agent in agents:
    agent.eval()
qmix.eval()

# Load test dataset
test_dataset = datasets.ImageFolder(root=r"C:\Users\sriau\Downloads\PreddRNN\PredRNN\data", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)  # Adjust batch size if needed

# Disable gradient computation for testing
correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)

        # Collect Q-values from each agent
        q_values = []
        for i, agent in enumerate(agents):
            agent_q_value = agent(images).squeeze(1)  # Shape: [batch_size]
            q_values.append(agent_q_value)

        # Stack Q-values and pass them through QMIX to get aggregated output
        q_values = torch.stack(q_values, dim=1)  # Shape: [batch_size, num_agents]
        qmix_output = qmix(q_values.to(device))  # Shape: [batch_size, 1]

        # Determine predicted class based on maximum Q-value across agents
        _, predicted = torch.max(q_values, dim=1)  # Get index of the highest Q-value per batch
        correct += (predicted == labels).sum().item()  # Count correct predictions
        total += labels.size(0)

# Calculate accuracy
accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
import os

# Create a directory to store the models if it doesn't exist
os.makedirs("saved_models", exist_ok=True)

# Save each CNN agent model
for i, agent in enumerate(agents):
    torch.save(agent.state_dict(), f"saved_models/cnn_agent_{i}.pth")

# Save the QMIX network model
torch.save(qmix.state_dict(), "saved_models/qmix_network.pth")

print("Models have been saved successfully.")
# Initialize the agents and QMIX network as before
agents = [CNNAgent().to(device) for _ in range(num_agents)]
qmix = QMIXNetwork(num_agents=num_agents).to(device)

# Load each CNN agent model
for i, agent in enumerate(agents):
    agent.load_state_dict(torch.load(f"saved_models/cnn_agent_{i}.pth"))

# Load the QMIX network model
qmix.load_state_dict(torch.load("saved_models/qmix_network.pth"))

print("Models have been loaded successfully.")