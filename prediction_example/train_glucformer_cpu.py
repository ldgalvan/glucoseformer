import torch
import torch.nn as nn
import numpy as np
import joblib
import time
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import math

# ========== CONFIG ==========
MODEL_SUFFIX = "baseline_2_5hr"
# ============================

# 1. Standard Transformer with Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_size=4, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.output(x)

# 2. Data Processing (NO IOB, NO PCA)
def load_and_preprocess(data_path):
    data = np.load(data_path)
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32)
    X_val = data["X_val"].astype(np.float32)
    y_val = data["y_val"].astype(np.float32)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)

    # Scaling
    X_scaler = StandardScaler().fit(X_train.reshape(-1, X_train.shape[-1]))
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))

    X_train = X_scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = X_scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test = X_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    y_train = y_scaler.transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
    y_val = y_scaler.transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

    joblib.dump(X_scaler, f'x_scaler_{MODEL_SUFFIX}.pkl')
    joblib.dump(y_scaler, f'y_scaler_{MODEL_SUFFIX}.pkl')

    print(f"Final shapes: X_train{X_train.shape}, X_val{X_val.shape}, X_test{X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test

# 3. Training Function
def train_model(model, train_loader, val_loader, device, n_epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    loss_fn = nn.L1Loss()
    best_loss = float('inf')
    patience = 7
    trigger_times = 0
    
    train_losses = []
    val_losses = []
    epoch_times = []

    for epoch in range(n_epochs):
        start_time = time.time()
        model.train()
        epoch_train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)[:, -12:, :].squeeze(-1)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * xb.size(0)
        
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)[:, -12:, :].squeeze(-1)
                loss = loss_fn(pred, yb)
                epoch_val_loss += loss.item() * xb.size(0)
        
        epoch_val_loss /= len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)

        epoch_duration = time.time() - start_time
        epoch_times.append(epoch_duration)
        print(f"Epoch {epoch+1:3d} | Train MAE: {epoch_train_loss:.4f} | Val MAE: {epoch_val_loss:.4f} | Time: {epoch_duration:.2f}s")

        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            trigger_times = 0
            torch.save(model.state_dict(), f'best_model_{MODEL_SUFFIX}.pth')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return train_losses, val_losses, epoch_times

def save_artifacts(train_losses, val_losses, epoch_times, suffix):
    np.savez(f'loss_curves_{suffix}.npz', train_losses=train_losses, val_losses=val_losses)
    np.savez(f'epoch_times_{suffix}.npz', epoch_times=epoch_times)

def calculate_metrics(y_true_scaled, y_pred_scaled, y_scaler):
    y_true = y_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    epsilon = 1e-8
    ape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return rmse, ape

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess("cgm_sequences.npz")
    
    # Create DataLoaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        ), 
        batch_size=batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        ), 
        batch_size=batch_size
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32)
        ), 
        batch_size=batch_size
    )
    
    # Initialize model
    input_size = X_train.shape[2]
    model = TransformerModel(input_size=input_size).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    train_losses, val_losses, epoch_times = train_model(model, train_loader, val_loader, device)
    save_artifacts(train_losses, val_losses, epoch_times, MODEL_SUFFIX)

    # Print average time/epoch
    avg_epoch_time = np.mean(epoch_times)
    print(f"\nAverage time per epoch: {avg_epoch_time:.2f}s")

    # Evaluate
    model.load_state_dict(torch.load(f'best_model_{MODEL_SUFFIX}.pth', map_location=device))
    model.eval()
    y_scaler = joblib.load(f'y_scaler_{MODEL_SUFFIX}.pkl')
    
    loss_fn = nn.L1Loss()
    all_preds = []
    all_targets = []
    test_loss = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)[:, -12:, :].squeeze(-1)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
            test_loss += loss_fn(pred, yb).item() * xb.size(0)
    
    test_loss /= len(test_loader.dataset)
    y_pred_scaled = np.concatenate(all_preds)
    y_true_scaled = np.concatenate(all_targets)
    rmse, ape = calculate_metrics(y_true_scaled, y_pred_scaled, y_scaler)
    
    print(f"\nFinal Metrics ({MODEL_SUFFIX}):")
    print(f"Test MAE (mg/dL): {test_loss * y_scaler.scale_[0]:.1f}")
    print(f"Test RMSE (mg/dL): {rmse:.1f}")
    print(f"Test Average Percentage Error: {ape:.1f}%")

if __name__ == "__main__":
    main()
