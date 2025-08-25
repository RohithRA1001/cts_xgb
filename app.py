from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import pickle
import pandas as pd

# -------------------------------
# Input Schema (features used in training)
# -------------------------------
class InputData(BaseModel):
    classification: str
    code: str
    implanted: str
    name_device: str
    name_manufacturer: str
    # optional: risk_class, determined_cause, reason can be left out and defaults used

# -------------------------------
# Load Model & Encoders
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "xgb_model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        label_encoders = pickle.load(f)
    print("✅ Model and encoders loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model/encoders: {e}")
    model, label_encoders = None, None

# -------------------------------
# Features used in training
# -------------------------------
TRAIN_FEATURES = [
    "classification", "code", "implanted", "name_device", "name_manufacturer",
    "risk_class", "determined_cause", "reason"
]

# -------------------------------
# Compute defaults for optional columns
# -------------------------------
# numeric: 0, categorical: mode
defaults = {}
for col in TRAIN_FEATURES:
    if col in label_encoders:
        le = label_encoders[col]
        defaults[col] = int(pd.Series(le.transform(le.classes_)).mode()[0])
    else:
        defaults[col] = 0  # numeric default

# -------------------------------
# Encode Input Row
# -------------------------------
def encode_input(data: InputData):
    row = {}

    # Fill user-provided features
    for col, val in data.dict().items():
        if col in label_encoders:
            le = label_encoders[col]
            if val in le.classes_:
                row[col] = int(le.transform([val])[0])
            else:
                # fallback to mode
                row[col] = defaults[col]
        else:
            try:
                row[col] = float(val)
            except:
                row[col] = defaults.get(col, 0)

    # Fill missing features with defaults
    for col in TRAIN_FEATURES:
        if col not in row:
            row[col] = defaults[col]

    # Convert to DataFrame
    df = pd.DataFrame([row])

    # Force all label-encoded columns to int
    for c in label_encoders:
        if c in df.columns:
            df[c] = df[c].astype(int)

    # Force remaining object columns to float
    for c in df.columns:
        if df[c].dtype == 'object':
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(float)

    # Ensure columns are in the same order as training
    df = df[TRAIN_FEATURES]
    return df

# -------------------------------
# Routes
# -------------------------------
@app.get("/")
def home():
    return {"message": "FastAPI backend is running!"}

@app.post("/predict")
def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        input_df = encode_input(data)

        # Predict
        pred_class = model.predict(input_df)[0]
        pred_probs = model.predict_proba(input_df)[0]

        # Decode class if label-encoded
        target_encoder = label_encoders.get("action_classification")  # adjust if different
        if target_encoder:
            try:
                pred_class_name = target_encoder.inverse_transform([int(pred_class)])[0]
            except Exception:
                pred_class_name = str(pred_class)
        else:
            pred_class_name = str(pred_class)

        return {
            "input": data.dict(),
            "prediction_class": pred_class_name,
            "prediction_id": int(pred_class),
            "class_probabilities": {
                str(cls): float(p) for cls, p in zip(model.classes_, pred_probs)
            },
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
