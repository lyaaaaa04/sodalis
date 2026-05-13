from fastapi import FastAPI
from pydantic import BaseModel
import datetime
import joblib
import pandas as pd
import numpy as np
import torch
torch.set_num_threads(2)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer

from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================================
# FASTAPI
# =========================================

app = FastAPI(title="Credit Scoring API")

# =========================================
# CUSTOM LAYER
# =========================================

class FeatureWeightingLayer(Layer):
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1],),
            initializer='ones',
            trainable=True
        )

    def call(self, inputs):
        return inputs * self.w

    def get_config(self):
        config = super().get_config()
        return config

# =========================================
# CUSTOM LOSS
# =========================================

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, name='focal_loss'):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        p_t = (y_true * y_pred + (1 - y_true) * (1 - y_pred))
        alpha_factor = (y_true * self.alpha + (1 - y_true) * (1 - self.alpha))
        modulating_factor = tf.pow((1.0 - p_t), self.gamma)

        loss = (alpha_factor * modulating_factor * bce)

        return tf.reduce_mean(loss)

# =========================================
# LOAD MODEL
# =========================================

scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

model = keras.models.load_model(
    "best_model.keras",
    custom_objects={
        "FeatureWeightingLayer": FeatureWeightingLayer,
        "FocalLoss": FocalLoss
    }
)

print("Credit scoring model loaded successfully!")

# =========================================
# DEVICE
# =========================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =========================================
# LOAD TINYLLAMA
# =========================================

LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading TinyLlama model...")

llm_tokenizer = AutoTokenizer.from_pretrained(
    LLM_MODEL
)

llm_tokenizer.pad_token = (
    llm_tokenizer.eos_token
)

llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.float32
).to(device)

llm_model.eval()

print("TinyLlama loaded successfully!")

# =========================================
# INPUT SCHEMA
# =========================================

class InputData(BaseModel):
    Home_Ownership: str
    Loan_Purpose: str
    Payment_History: int
    Previous_Loan: str
    Parental_Income_IDR_Monthly: float
    Loan_Amount_IDR: int
    Working_Student: str
    Course_Credits: int
    Liability: int
    Attendance: float
    Grade_Average: float
    Parent_Job: str
    Residence_Type: str

# =========================================
# FEATURES
# =========================================

numerical_features = [
    'Payment_History',
    'Loan_Amount_IDR',
    'Course_Credits',
    'Liability',
    'Attendance',
    'Grade_Average',
    'DSR'
]

categorical_features = [
    'Home_Ownership',
    'Loan_Purpose',
    'Previous_Loan',
    'Working_Student',
    'Parent_Job',
    'Residence_Type'
]

# =========================================
# HOME
# =========================================

@app.get("/")
def home():
    return {"message": "API is running"}

# =========================================
# HEALTH
# =========================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": type(model).__name__,
        "time": datetime.datetime.utcnow().isoformat() + "Z"
    }

# =========================================
# FACTOR EXTRACTION
# =========================================

def extract_explanation_factors(row):

    ekonomi_reasons, pendidikan_reasons, credit_reasons = [], [], []
    ekonomi_score, pendidikan_score, credit_score_rule = 0, 0, 0

    home = row['Home_Ownership']

    if home == 'rent':
        ekonomi_score += 1
        ekonomi_reasons.append("keluarga masih menggunakan rumah sewa")
    elif home == 'mortgage':
        ekonomi_score += 2
        ekonomi_reasons.append("keluarga memiliki rumah dengan sistem cicilan")
    elif home == 'own':
        ekonomi_score += 3
        ekonomi_reasons.append("keluarga memiliki rumah pribadi")

    job = row['Parent_Job']

    if job in ['pns', 'teacher', 'nurse']:
        ekonomi_score += 3
        ekonomi_reasons.append("pekerjaan orang tua tergolong stabil")
    elif job in ['karyawan swasta', 'wirausaha', 'pedagang']:
        ekonomi_score += 2
        ekonomi_reasons.append("orang tua memiliki pekerjaan dengan pendapatan cukup baik")
    else:
        ekonomi_score += 1
        ekonomi_reasons.append("pekerjaan orang tua memiliki tingkat kestabilan rendah")

    dep = row['Liability']

    if dep <= 1:
        ekonomi_score += 3
        ekonomi_reasons.append("jumlah tanggungan keluarga relatif rendah")
    elif dep <= 3:
        ekonomi_score += 2
        ekonomi_reasons.append("jumlah tanggungan keluarga berada pada kategori sedang")
    else:
        ekonomi_score += 1
        ekonomi_reasons.append("jumlah tanggungan keluarga cukup besar")
    
    residence = row['Residence_Type']

    if residence == 'urban':
        ekonomi_score += 2
        ekonomi_reasons.append("keluarga tinggal di wilayah perkotaan")
    elif residence == 'rural':
        ekonomi_score += 1
        ekonomi_reasons.append("keluarga tinggal di wilayah pedesaan")

    gpa = row['Grade_Average']

    if gpa < 2:
        pendidikan_score += 1
        pendidikan_reasons.append("performa akademik mahasiswa masih rendah")
    elif gpa <= 3:
        pendidikan_score += 2
        pendidikan_reasons.append("performa akademik mahasiswa cukup baik")
    else:
        pendidikan_score += 3
        pendidikan_reasons.append("mahasiswa memiliki performa akademik yang sangat baik")

    att = row['Attendance']

    if att < 60:
        pendidikan_score += 1
        pendidikan_reasons.append("tingkat kehadiran kuliah tergolong rendah")
    elif att <= 80:
        pendidikan_score += 2
        pendidikan_reasons.append("tingkat kehadiran kuliah cukup baik")
    else:
        pendidikan_score += 3
        pendidikan_reasons.append("tingkat kehadiran kuliah sangat baik")

    sks = row['Course_Credits']

    if sks < 20:
        pendidikan_score += 1
        pendidikan_reasons.append("jumlah SKS yang diambil masih rendah")
    elif sks <= 22:
        pendidikan_score += 2
        pendidikan_reasons.append("jumlah SKS yang diambil cukup optimal")
    else:
        pendidikan_score += 3
        pendidikan_reasons.append("mahasiswa mengambil beban SKS yang tinggi")

    working = row['Working_Student']

    if working == 'bekerja_karena_butuh':
        pendidikan_score += 2
        pendidikan_reasons.append("mahasiswa bekerja karena kebutuhan ekonomi")
    else:
        pendidikan_score += 1
        pendidikan_reasons.append("mahasiswa bekerja secara opsional")

    history = row['Payment_History']

    if history >= 72:
        credit_score_rule += 3
        credit_reasons.append("riwayat pembayaran pinjaman tergolong sangat baik")
    elif history >= 48:
        credit_score_rule += 2
        credit_reasons.append("riwayat pembayaran pinjaman cukup baik")
    else:
        credit_score_rule += 1
        credit_reasons.append("riwayat pembayaran pinjaman kurang baik")

    prev = row['Previous_Loan']

    if prev == 'yes':
        credit_score_rule += 2
        credit_reasons.append("mahasiswa memiliki pengalaman pinjaman sebelumnya")
    else:
        credit_score_rule += 1
        credit_reasons.append("mahasiswa belum memiliki pengalaman pinjaman sebelumnya")

    loan = row['Loan_Amount_IDR']

    if loan <= 10000000:
        credit_score_rule += 3
        credit_reasons.append("jumlah pinjaman yang diajukan relatif rendah")
    elif loan <= 25000000:
        credit_score_rule += 2
        credit_reasons.append("jumlah pinjaman yang diajukan berada pada kategori sedang")
    else:
        credit_score_rule += 1
        credit_reasons.append("jumlah pinjaman yang diajukan cukup besar")

    purpose = row['Loan_Purpose']

    if purpose == 'education':
        credit_score_rule += 3
        credit_reasons.append("pinjaman digunakan untuk kebutuhan pendidikan")
    elif purpose == 'medical':
        credit_score_rule += 3
        credit_reasons.append("pinjaman digunakan untuk kebutuhan medis")
    elif purpose == 'venture':
        credit_score_rule += 2
        credit_reasons.append("pinjaman digunakan untuk pengembangan usaha")
    else:
        credit_score_rule += 1
        credit_reasons.append("pinjaman digunakan untuk kebutuhan pribadi")

    return {
        "ekonomi_score": ekonomi_score,
        "pendidikan_score": pendidikan_score,
        "credit_score_rule": credit_score_rule,
        "ekonomi_reasons": ekonomi_reasons,
        "pendidikan_reasons": pendidikan_reasons,
        "credit_reasons": credit_reasons
    }

# =========================================
# BUILD EXPLANATION
# =========================================

def build_structured_explanation(status, factors):

    ekonomi = ", ".join(factors["ekonomi_reasons"][:2])
    pendidikan = ", ".join(factors["pendidikan_reasons"][:2])
    credit = ", ".join(factors["credit_reasons"][:2])

    explanation = (
        f"Mahasiswa diprediksi {status.lower()} menerima pinjaman "
        f"berdasarkan hasil evaluasi pada aspek ekonomi, pendidikan, dan credit scoring. "
        f"Pada aspek ekonomi, {ekonomi}. "
        f"Pada aspek pendidikan, {pendidikan}. "
        f"Pada aspek credit scoring, {credit}."
    )

    return explanation

# =========================================
# PREDICT
# =========================================

@app.post("/predict")
def predict(data: InputData):

    # DSR dihitung dulu
    dsr = (
        data.Loan_Amount_IDR /
        (data.Parental_Income_IDR_Monthly + 1e-7)
    )

    raw_input_data = pd.DataFrame({
        'Home_Ownership': [data.Home_Ownership],
        'Loan_Purpose': [data.Loan_Purpose],
        'Payment_History': [data.Payment_History],
        'Previous_Loan': [data.Previous_Loan],
        'Parental_Income_IDR_Monthly': [
            data.Parental_Income_IDR_Monthly
        ],
        'Loan_Amount_IDR': [data.Loan_Amount_IDR],
        'Working_Student': [data.Working_Student],
        'Course_Credits': [data.Course_Credits],
        'Liability': [data.Liability],
        'Attendance': [data.Attendance],
        'Grade_Average': [data.Grade_Average],
        'Parent_Job': [data.Parent_Job],
        'Residence_Type': [data.Residence_Type],
        'DSR': [dsr]
    })

    input_data = raw_input_data.copy()

    # normalisasi text
    for col in categorical_features:
        input_data[col] = (
            input_data[col]
            .astype(str)
            .str.lower()
            .str.strip()
        )

    # IMPORTANT:
    # log1p dilakukan setelah DSR dihitung
    input_data['Loan_Amount_IDR'] = np.log1p(
        input_data['Loan_Amount_IDR']
    )

    original_data = raw_input_data.copy()

    # scaling
    input_data[numerical_features] = scaler.transform(
        input_data[numerical_features]
    )

    # encoding
    for col in categorical_features:

        le = label_encoders[col]

        mapping = dict(
            zip(
                le.classes_,
                le.transform(le.classes_)
            )
        )

        input_data[col] = (
            input_data[col]
            .map(mapping)
            .fillna(0)
            .astype(int)
        )

    # format model input
    input_dict = {}

    for col in categorical_features:
        input_dict[f"{col}_input"] = (
            input_data[col]
            .values
            .reshape(-1, 1)
        )

    input_dict["numerical_input"] = (
        input_data[numerical_features]
        .values
        .astype("float32")
    )

    # prediction
    prediction = model.predict(
        input_dict,
        verbose=0
    )

    prob = float(prediction[0][0])

    credit_score = 850 - (prob * 550)

    if credit_score >= 600:
        predicted_label = "Layak"
    else:
        predicted_label = "Tidak Layak"

    # explanation factors
    factors = extract_explanation_factors(
        original_data.iloc[0]
    )

    base_explanation = (
        build_structured_explanation(
            predicted_label,
            factors
        )
    )

    # prompt
    prompt = f"""
Fix the grammar of the following sentence.

Rules:
- Do not add new information
- Do not explain anything
- Keep the exact meaning
- Keep the same facts
- Return only one sentence

Sentence:
{base_explanation}

Fixed sentence:
"""

    # tokenize
    inputs = llm_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    # generate
    with torch.no_grad():

        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=45,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=llm_tokenizer.eos_token_id,
            eos_token_id=llm_tokenizer.eos_token_id
        )

    # decode
    response = llm_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    response = response.replace(prompt, "").strip()
    response = response.split("\n")[0].strip()

    bad_patterns = [
        "kepala dinas",
        "tidak ada yang anda boleh",
        "bullet point",
        "- ",
        "1.",
        "2."
    ]

    is_bad = any(p in response.lower() for p in bad_patterns)

    if len(response) < 30 or is_bad:
        explanation = base_explanation
    else:
        explanation = response

    response_lower = response.lower()

    required_keywords = [
        "student",
        "loan",
        "economic",
        "education",
        "credit"
    ]

    missing_keywords = any(
        k not in response_lower
        for k in required_keywords
    )

    if (
        len(response) < 40
        or is_bad
        or missing_keywords
    ):
        explanation = base_explanation
    else:
        explanation = response

    # return api
    return {
        "prediction": predicted_label,
        "credit_score": round(credit_score, 2),
        "explanation": explanation,
        "scores": {
            "ekonomi_score": factors["ekonomi_score"],
            "pendidikan_score": factors["pendidikan_score"],
            "credit_score": factors["credit_score_rule"]
        }
    }