import joblib
import pandas as pd
import torch
torch.set_num_threads(2)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
from transformers import AutoTokenizer, AutoModelForCausalLM

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
# LOAD CREDIT SCORING MODEL
# =========================================

scaler = joblib.load('scaler2.pkl')
label_encoders = joblib.load('encoders2.pkl')
target_encoder = joblib.load("target_encoder2.pkl")

model = keras.models.load_model(
    "best_model2.keras",
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
# LOAD TINYLLAMA MODEL
# =========================================

LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading TinyLlama model...")

llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

llm_tokenizer.pad_token = llm_tokenizer.eos_token

llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.float32
).to(device)

llm_model.eval()

print("TinyLlama loaded successfully!")

# =========================================
# INPUT DATA
# =========================================

Home_Ownership = input("Enter home ownership (rent, own, mortgage): ").lower().strip()
Loan_Purpose = input("Enter loan purpose (education, medical, venture, personal): ").lower().strip()
Payment_History = int(input("Enter payment history: "))
Previous_Loan = input("Enter previous loan (yes, no): ").lower().strip()
Parental_Income_IDR_Monthly = float(input("Enter parental income: "))
Loan_Amount_IDR = int(input("Enter loan amount: "))
Working_Student = input("Working student (bekerja_karena_butuh, bekerja_optional): ").lower().strip()
Course_Credits = int(input("Course credits: "))
Liability = int(input("Liability: "))
Attendance = float(input("Attendance: "))
Grade_Average = float(input("Grade average: "))
Parent_Job = input("Parent job: ").lower().strip()
Residence_Type = input("Residence type (urban, rural): ").lower().strip()

DSR = Loan_Amount_IDR / (Parental_Income_IDR_Monthly + 1e-7)

# =========================================
# DATAFRAME
# =========================================

raw_input_data = pd.DataFrame({
    'Home_Ownership': [Home_Ownership],
    'Loan_Purpose': [Loan_Purpose],
    'Payment_History': [Payment_History],
    'Previous_Loan': [Previous_Loan],
    'Parental_Income_IDR_Monthly': [Parental_Income_IDR_Monthly],
    'Loan_Amount_IDR': [Loan_Amount_IDR],
    'Working_Student': [Working_Student],
    'Course_Credits': [Course_Credits],
    'Liability': [Liability],
    'Attendance': [Attendance],
    'Grade_Average': [Grade_Average],
    'Parent_Job': [Parent_Job],
    'Residence_Type': [Residence_Type],
    'DSR': [DSR]
})

input_data = raw_input_data.copy()

# =========================================
# FEATURES
# =========================================

numerical_features = [
    'Payment_History', 'Parental_Income_IDR_Monthly',
    'Loan_Amount_IDR', 'Course_Credits',
    'Liability', 'Attendance',
    'Grade_Average', 'DSR'
]

categorical_features = [
    'Home_Ownership', 'Loan_Purpose',
    'Previous_Loan', 'Working_Student',
    'Parent_Job', 'Residence_Type'
]

# =========================================
# NORMALIZATION
# =========================================

for col in categorical_features:
    input_data[col] = input_data[col].astype(str).str.lower().str.strip()

original_data = input_data.copy()

# =========================================
# SCALING
# =========================================

input_data[numerical_features] = scaler.transform(input_data[numerical_features])

# =========================================
# ENCODING
# =========================================

for col in categorical_features:
    le = label_encoders[col]
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    input_data[col] = input_data[col].map(mapping).fillna(0).astype(int)

# =========================================
# FORMAT MODEL INPUT
# =========================================

input_dict = {}

for col in categorical_features:
    input_dict[f"{col}_input"] = input_data[col].values.reshape(-1, 1)

input_dict["numerical_input"] = input_data[numerical_features].values.astype('float32')

# =========================================
# PREDICTION
# =========================================

prediction = model.predict(input_dict, verbose=0)

prob = float(prediction[0][0])
predicted_class = 1 if prob >= 0.5 else 0
predicted_label = target_encoder.inverse_transform([predicted_class])[0]

credit_score = 300 + ((1 - prob) * 550)

# =========================================
# EXPLANATION GENERATIVE AI
# =========================================

def extract_explanation_factors(row):

    ekonomi_reasons, pendidikan_reasons, credit_reasons = [], [], []
    ekonomi_score, pendidikan_score, credit_score_rule = 0, 0, 0

    income = row['Parental_Income_IDR_Monthly']

    if income < 19000000:
        ekonomi_score += 1
        ekonomi_reasons.append("penghasilan orang tua tergolong rendah")
    elif income <= 27000000:
        ekonomi_score += 2
        ekonomi_reasons.append("penghasilan orang tua berada pada kategori menengah")
    elif income <= 37000000:
        ekonomi_score += 3
        ekonomi_reasons.append("penghasilan orang tua tergolong baik")
    else:
        ekonomi_score += 4
        ekonomi_reasons.append("penghasilan orang tua sangat stabil")

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
# GET FACTORS
# =========================================

factors = extract_explanation_factors(original_data.iloc[0])

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

base_explanation = build_structured_explanation(
    predicted_label,
    factors
)

# =========================================
# PROMPT
# =========================================

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

# =========================================
# TOKENIZATION
# =========================================

inputs = llm_tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=512
).to(device)

print("\nGenerating explanation...\n")

# =========================================
# GENERATE
# =========================================

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

# =========================================
# DECODE
# =========================================

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

# =========================================
# FINAL RESULT
# =========================================

print("\nHASIL AKHIR")

print(f"\nKelas         : {predicted_label}")
print(f"Probability     : {prob*100:.2f}%")
print(f"Credit Score    : {credit_score:.2f}")

print("\nSCORE PER ASPEK")
print(f"Ekonomi Score     : {factors['ekonomi_score']}")
print(f"Pendidikan Score  : {factors['pendidikan_score']}")
print(f"Credit Score      : {factors['credit_score_rule']}")

print("\nEXPLANATION GENERATIVE AI")
print(explanation)