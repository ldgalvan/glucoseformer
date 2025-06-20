import torch
import numpy as np
import joblib
from fastapi import FastAPI, UploadFile, File
import io
from train_glucformer_cpu import TransformerModel

# ----------------------
# CONFIG
# ----------------------
MODEL_SUFFIX = "baseline_5hr"
SEQUENCE_PATH = "one_sequence.npy"
MODEL_PATH = f"best_model_{MODEL_SUFFIX}.pth"
X_SCALER_PATH = f"x_scaler_{MODEL_SUFFIX}.pkl"
Y_SCALER_PATH = f"y_scaler_{MODEL_SUFFIX}.pkl"

# ----------------------
# MODEL SETUP
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(input_size=4).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

x_scaler = joblib.load(X_SCALER_PATH)
y_scaler = joblib.load(Y_SCALER_PATH)

# ----------------------
# FASTAPI APP
# ----------------------
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(None)):
    if file:
        contents = await file.read()
        sequence = np.load(io.BytesIO(contents))
    else:
        sequence = np.load(SEQUENCE_PATH)

    assert sequence.shape == (60, 4), "Input sequence must have shape (60, 4)"

    sequence_scaled = x_scaler.transform(sequence)
    sequence_scaled = torch.tensor(sequence_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output_scaled = model(sequence_scaled)[:, -12:, :].squeeze(0).squeeze(-1).cpu().numpy()

    output = y_scaler.inverse_transform(output_scaled.reshape(-1, 1)).flatten()
    return {"predicted_cgm": output.tolist()}
