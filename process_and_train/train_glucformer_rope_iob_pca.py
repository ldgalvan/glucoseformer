import torch
import torch.nn as nn
import numpy as np
import joblib
import time
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import math
from rotary_embedding_torch import RotaryEmbedding


MODEL_SUFFIX = "2_5hr_ropepcaiob"  


# 1. RoPE-enabled Multihead Attention and Transformer
class MultiheadSelfAttentionWithRoPE(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rotary_emb = RotaryEmbedding(dim=self.head_dim)

    def forward(self, x):
        B, S, D = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, S, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(attn_output)

class TransformerEncoderLayerWithRoPE(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadSelfAttentionWithRoPE(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src):
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerModel(nn.Module):
    def __init__(self, input_size=3, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_size, d_model)
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithRoPE(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)

# 2. Data Processing with PCA
def load_and_preprocess(data_path, pca_components=None, suffix=MODEL_SUFFIX):
    data = np.load(data_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    def compute_iob(bolus_sequence, action_minutes=240, interval=5):
        iob = np.zeros_like(bolus_sequence)
        for t in range(len(bolus_sequence)):
            for past in range(max(0, t - action_minutes//interval), t+1):
                mins_ago = (t - past) * interval
                if mins_ago <= action_minutes:
                    iob[t] += bolus_sequence[past] * (1 - mins_ago/action_minutes)
        return iob

    def add_iob(X):
        X_new = np.zeros((X.shape[0], X.shape[1], X.shape[2] + 1))
        X_new[:, :, :-1] = X
        for i in range(X.shape[0]):
            X_new[i, :, -1] = compute_iob(X[i, :, 2] + X[i, :, 3])
        return X_new

    # Add IOB feature
    X_train = add_iob(X_train)
    X_val = add_iob(X_val)
    X_test = add_iob(X_test)

    # Flatten for scaling
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])

    # Scale features
    X_scaler = StandardScaler().fit(X_train_flat)
    X_train = X_scaler.transform(X_train_flat).reshape(X_train.shape)
    X_val = X_scaler.transform(X_val_flat).reshape(X_val.shape)
    X_test = X_scaler.transform(X_test_flat).reshape(X_test.shape)

    # Apply PCA if specified
    if pca_components:
        pca = PCA(n_components=pca_components)
        X_train = pca.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape[0], X_train.shape[1], -1)
        X_val = pca.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape[0], X_val.shape[1], -1)
        X_test = pca.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape[0], X_test.shape[1], -1)
        joblib.dump(pca, f'pca_transformer_{suffix}.pkl')
        print(f"Applied PCA: {X_train.shape[-1]} components")

    # Scale targets
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    y_train = y_scaler.transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
    y_val = y_scaler.transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

    joblib.dump(X_scaler, f'x_scaler_{suffix}.pkl')
    joblib.dump(y_scaler, f'y_scaler_{suffix}.pkl')

    print(f"Final shapes: X_train{X_train.shape}, X_val{X_val.shape}, X_test{X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test

# 3. Training with Time/Epoch Tracking
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
            torch.save(model.state_dict(), 'best_transformer_ropepcaiob.pth')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return train_losses, val_losses, epoch_times

# 4. Metrics and Saving Functions
def calculate_metrics(y_true_scaled, y_pred_scaled, y_scaler):
    y_true = y_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    epsilon = 1e-8
    ape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return rmse, ape

# 4. Metrics and Saving (Modified)
def save_artifacts(train_losses, val_losses, epoch_times, model_suffix):
    """Save all artifacts with dynamic suffix."""
    np.savez(f'loss_curves_{model_suffix}.npz', 
             train_losses=train_losses, 
             val_losses=val_losses)
    np.savez(f'epoch_times_{model_suffix}.npz', 
             epoch_times=epoch_times)

# 5. Main Execution with Custom Suffix
def main():

    
    # Config
    PCA_COMPONENTS = 3  # Set to None to disable PCA
    torch.manual_seed(42)
    np.random.seed(42)

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess(
        "cgm_sequences.npz", 
        pca_components=PCA_COMPONENTS
    )
    
    # Create DataLoaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), 
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32), 
        torch.tensor(y_val, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32), 
        torch.tensor(y_test, dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    input_size = X_train.shape[2]
    model = TransformerModel(
        input_size=input_size,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.2
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    train_losses, val_losses, epoch_times = train_model(model, train_loader, val_loader, device)
    save_artifacts(train_losses, val_losses, epoch_times, MODEL_SUFFIX)

    # Save model
    torch.save(model.state_dict(), f'best_model_{MODEL_SUFFIX}.pth')

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