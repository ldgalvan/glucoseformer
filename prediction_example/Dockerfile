# Base image (no GPU needed)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy only necessary files
COPY predict_sequence.py train_glucformer_cpu.py one_sequence.npy ./
COPY best_model_baseline_5hr.pth x_scaler_baseline_5hr.pkl y_scaler_baseline_5hr.pkl ./

# Install dependencies
RUN pip install --no-cache-dir torch==2.5.1 joblib numpy fastapi uvicorn scikit-learn python-multipart

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "predict_sequence:app", "--host", "0.0.0.0", "--port", "8000"]

