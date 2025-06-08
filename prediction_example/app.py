from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import torch
import numpy as np
import joblib
import os
from train_glucformer_cpu import TransformerModel

app = FastAPI()

# ----------------------
# CONFIG
# ----------------------
MODEL_SUFFIX = "baseline_5hr"
DEFAULT_SEQUENCE = "one_sequence.npy"
MODEL_PATH = f"best_model_{MODEL_SUFFIX}.pth"
X_SCALER_PATH = f"x_scaler_{MODEL_SUFFIX}.pkl"
Y_SCALER_PATH = f"y_scaler_{MODEL_SUFFIX}.pkl"

# ----------------------
# LOAD MODEL & SCALERS ONCE
# ----------------------
device = torch.device("cpu")
model = TransformerModel(input_size=4).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

x_scaler = joblib.load(X_SCALER_PATH)
y_scaler = joblib.load(Y_SCALER_PATH)

# ----------------------
# REQUEST MODEL
# ----------------------
class FilePathRequest(BaseModel):
    file_path: Optional[str] = None

@app.post("/predict")
def predict(request: FilePathRequest):
    file_path = request.file_path or DEFAULT_SEQUENCE

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    try:
        sequence = np.load(file_path)
        if sequence.shape != (60, 4):
            raise ValueError(f"Expected shape (60, 4), got {sequence.shape}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading file: {e}")

    # Preprocess
    sequence_scaled = x_scaler.transform(sequence)
    sequence_scaled = torch.tensor(sequence_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output_scaled = model(sequence_scaled)[:, -12:, :].squeeze(0).squeeze(-1).cpu().numpy()

    output = y_scaler.inverse_transform(output_scaled.reshape(-1, 1)).flatten()
    return {"predicted_cgm": output.tolist()}
