import torch
import numpy as np
import joblib
#from train_glucformer import TransformerModel  # Your model definition file


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "process_and_train")))

from train_glucformer import TransformerModel


# ----------------------
# CONFIG
# ----------------------
MODEL_SUFFIX = "baseline_5hr"
SEQUENCE_PATH = "one_sequence.npy"
MODEL_PATH = f"best_model_{MODEL_SUFFIX}.pth"
X_SCALER_PATH = f"x_scaler_{MODEL_SUFFIX}.pkl"
Y_SCALER_PATH = f"y_scaler_{MODEL_SUFFIX}.pkl"

# ----------------------
# LOAD INPUT
# ----------------------
sequence = np.load(SEQUENCE_PATH)  # shape: (60, 4)
assert sequence.shape == (60, 4), "Input sequence must have shape (60, 4)"

# ----------------------
# LOAD SCALERS AND MODEL
# ----------------------
x_scaler = joblib.load(X_SCALER_PATH)
y_scaler = joblib.load(Y_SCALER_PATH)

sequence_scaled = x_scaler.transform(sequence)
sequence_scaled = torch.tensor(sequence_scaled, dtype=torch.float32).unsqueeze(0)  # (1, 60, 4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(input_size=4).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ----------------------
# INFERENCE
# ----------------------
with torch.no_grad():
    sequence_scaled = sequence_scaled.to(device)
    output_scaled = model(sequence_scaled)[:, -12:, :].squeeze(0).squeeze(-1).cpu().numpy()

output = y_scaler.inverse_transform(output_scaled.reshape(-1, 1)).flatten()

# ----------------------
# OUTPUT
# ----------------------
print("Predicted CGM values (mg/dL):")
print(output.tolist())

