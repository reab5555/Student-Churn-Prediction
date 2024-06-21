import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from tabulate import tabulate
import json

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the data
X = pd.read_csv("X_resampled.csv").values
y = pd.read_csv("y_resampled.csv").squeeze().values


# Load layer configurations and learning rates
with open("layer_configurations.json", "r") as f:
    configs = json.load(f)

layer_configs = configs["layer_configs"]
learning_rates = configs["learning_rates"]

# Define the neural network model with flexible architecture
class FlexibleAlzheimerNet(nn.Module):
    def __init__(self, input_dim, layer_sizes):
        super(FlexibleAlzheimerNet, self).__init__()
        layers = []
        prev_size = input_dim
        for size in layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Function to train the model with early stopping
def train_model(model, X_train, y_train, X_val, y_val, epochs=250, batch_size=128, patience=2, learning_rate=0.001,
                fold=1):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    counter = 0

    progress_bar = trange(epochs, desc=f"Fold {fold} Training", leave=False, colour='blue')
    for epoch in progress_bar:
        model.train()
        epoch_train_loss = 0
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= (len(X_train) // batch_size)
        train_losses.append(epoch_train_loss)

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val.unsqueeze(1)).item()
            val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                progress_bar.write(f"Early stopping at epoch {epoch + 1}")
                break

        progress_bar.set_postfix(
            {"Epoch": epoch + 1, "Train Loss": epoch_train_loss, "Val Loss": val_loss, "LR": learning_rate})

    return train_losses, val_losses

# Function to plot learning curves
def plot_learning_curves(all_train_losses, all_val_losses):
    plt.figure(figsize=(12, 6))

    for i, (train_losses, val_losses) in enumerate(zip(all_train_losses, all_val_losses)):
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, linestyle='--', label=f'Train (Fold {i + 1})')
        plt.plot(epochs, val_losses, label=f'Validation (Fold {i + 1})')

    plt.title('Learning Curves for All Folds')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Customize x-axis
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Customize y-axis
    y_min = min(min(losses) for losses in all_train_losses + all_val_losses)
    y_max = max(max(losses) for losses in all_train_losses + all_val_losses)
    plt.ylim(y_min - 0.1, y_max + 0.1)
    plt.gca().yaxis.set_major_locator(ticker.AutoLocator())
    plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Function to plot confusion matrix
def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Average Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Function to plot ROC curve
def plot_roc_curve(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

# Perform grid search with 8-fold cross-validation
skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
best_f1_score = 0
best_config = None
best_lr = None
best_train_losses = []
best_val_losses = []
best_y_true = []
best_y_pred = []
best_y_pred_proba = []

total_configs = len(layer_configs) * len(learning_rates)
progress_bar = tqdm(total=total_configs, desc="Grid Search", colour='green')

for layer_sizes in layer_configs:
    for lr in learning_rates:
        cv_f1_scores = []
        all_y_true = []
        all_y_pred = []
        all_y_pred_proba = []
        all_train_losses = []
        all_val_losses = []

        for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
            # Split the data
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Downsample training data
            rus = RandomUnderSampler(random_state=42)
            X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

            # Convert to tensors and move to device
            X_train = torch.FloatTensor(X_train_resampled).to(device)
            y_train = torch.FloatTensor(y_train_resampled).to(device)
            X_val = torch.FloatTensor(X_val).to(device)
            y_val = torch.FloatTensor(y_val).to(device)

            model = FlexibleAlzheimerNet(X_train.shape[1], layer_sizes).to(device)

            # Train the model
            train_losses, val_losses = train_model(model, X_train, y_train, X_val, y_val, learning_rate=lr, fold=fold)

            all_train_losses.append(train_losses)
            all_val_losses.append(val_losses)

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_preds = (val_outputs > 0.5).float().cpu().numpy()
                y_val_cpu = y_val.cpu().numpy()
                val_pred_proba = val_outputs.cpu().numpy()

            all_y_true.extend(y_val_cpu)
            all_y_pred.extend(val_preds)
            all_y_pred_proba.extend(val_pred_proba)

            f1 = f1_score(y_val_cpu, val_preds)
            cv_f1_scores.append(f1)

        mean_cv_f1_score = np.mean(cv_f1_scores)

        if mean_cv_f1_score > best_f1_score:
            best_f1_score = mean_cv_f1_score
            best_config = layer_sizes
            best_lr = lr
            best_train_losses = all_train_losses
            best_val_losses = all_val_losses
            best_y_true = all_y_true
            best_y_pred = all_y_pred
            best_y_pred_proba = all_y_pred_proba

        progress_bar.set_postfix({"Config": layer_sizes, "LR": lr, "Best F1": best_f1_score})
        progress_bar.update(1)

progress_bar.close()

# Plot learning curves for the best configuration
plot_learning_curves(best_train_losses, best_val_losses)

# Print average classification report for the best configuration
print("\nAverage Classification Report:")
classification_rep = classification_report(best_y_true, best_y_pred, output_dict=True)

# Convert classification report to table format
classification_table = []
for label, metrics in classification_rep.items():
    if isinstance(metrics, dict):
        row = [label]
        row.extend([metrics['precision'], metrics['recall'], metrics['f1-score'], metrics['support']])
        classification_table.append(row)
    else:
        row = [label, metrics]
        classification_table.append(row)

headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]

print(tabulate(classification_table, headers=headers, tablefmt="fancy_grid"))

# Plot average confusion matrix for the best configuration
avg_cm = confusion_matrix(best_y_true, best_y_pred, normalize='true')
plot_confusion_matrix(avg_cm)

# Plot ROC curve for the best configuration
plot_roc_curve(best_y_true, best_y_pred_proba)

print(f"\nBest configuration: {best_config} with learning rate: {best_lr} and F1 score: {best_f1_score:.4f}")

# Train final model with best configuration and learning rate
final_model = FlexibleAlzheimerNet(X.shape[1], best_config).to(device)
X_tensor = torch.FloatTensor(X).to(device)
y_tensor = torch.FloatTensor(y).to(device)
train_losses, val_losses = train_model(final_model, X_tensor, y_tensor, X_tensor, y_tensor, learning_rate=best_lr)

# Save the final model
torch.save(final_model.state_dict(), 'alzheimer_prediction_model.pth')
print("Final model saved as 'alzheimer_prediction_model.pth'")

