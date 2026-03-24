"""
train_model.py
==============
Huấn luyện Logistic Regression (mô hình tốt nhất từ DoAn_II___.ipynb)
trên tập dữ liệu 2.5M Reviews và lưu model + vectorizer.

CÁCH DÙNG:
    python train_model.py --data /path/to/2.5m-reviews-dataset.csv
    python train_model.py --data /path/to/data.csv --sample 200000
"""

import argparse
import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Project-local imports
from utils.preprocessing import preprocess_text, score_to_sentiment

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
MODEL_DIR   = "models"
TFIDF_PATH  = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
MODEL_PATH  = os.path.join(MODEL_DIR, "logistic_regression.joblib")
LABEL_PATH  = os.path.join(MODEL_DIR, "label_encoder.joblib")
META_PATH   = os.path.join(MODEL_DIR, "model_meta.joblib")

TFIDF_PARAMS = dict(
    max_features=10_000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True,
)

LR_PARAMS = dict(
    max_iter=1000,
    C=1.0,
    solver='lbfgs',
    n_jobs=-1,
    random_state=42,
)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Train sentiment model")
    parser.add_argument("--data",   required=True,  help="Path to CSV dataset")
    parser.add_argument("--sample", type=int, default=None,
                        help="Use a random sample of N rows (default: full dataset)")
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── 1. Load data ────────────────────────────────────────────────────────
    print(f"\n[1/6] Đang đọc dữ liệu từ: {args.data}")
    t0 = time.time()
    df = pd.read_csv(args.data, encoding='latin1', low_memory=False,
                     dtype={'score': 'object'})
    df.columns = df.columns.str.lower().str.strip()
    print(f"      Đọc xong: {df.shape[0]:,} hàng ({time.time()-t0:.1f}s)")

    # ── 2. Score → Sentiment ────────────────────────────────────────────────
    print("\n[2/6] Gán nhãn Sentiment …")
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df = df[df['score'].between(1, 5)].copy()
    df['score'] = df['score'].astype(int)
    df['Sentiment'] = df['score'].apply(score_to_sentiment)

    # ── 3. Clean & preprocess ───────────────────────────────────────────────
    print("\n[3/6] Tiền xử lý văn bản …")
    df = df.dropna(subset=['review']).copy()
    t0 = time.time()
    df['review_processed'] = df['review'].apply(preprocess_text)
    df = df[df['review_processed'].str.len() > 0]
    df.drop_duplicates(subset=['review_processed', 'Sentiment'], inplace=True)
    print(f"      Xong: {df.shape[0]:,} hàng ({time.time()-t0:.1f}s)")

    # Optional sample
    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=42)
        print(f"      Dùng sample: {args.sample:,} hàng")

    # ── 4. Encode labels ────────────────────────────────────────────────────
    le = LabelEncoder()
    y = le.fit_transform(df['Sentiment'])
    print(f"\n[4/6] Nhãn: {le.classes_.tolist()}")
    print("      Phân phối:", dict(zip(*np.unique(y, return_counts=True))))

    # ── 5. TF-IDF + Split ───────────────────────────────────────────────────
    print("\n[5/6] TF-IDF vectorisation …")
    X_train, X_test, y_train, y_test = train_test_split(
        df['review_processed'], y, test_size=0.2, random_state=42, stratify=y
    )
    tfidf = TfidfVectorizer(**TFIDF_PARAMS)
    t0 = time.time()
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec  = tfidf.transform(X_test)
    print(f"      Xong ({time.time()-t0:.1f}s) | vocab={len(tfidf.vocabulary_):,}")

    # ── 6. Train LR ─────────────────────────────────────────────────────────
    print("\n[6/6] Huấn luyện Logistic Regression …")
    model = LogisticRegression(**LR_PARAMS)
    t0 = time.time()
    model.fit(X_train_vec, y_train)
    t_train = time.time() - t0
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='macro', zero_division=0)
    print(f"\n  Accuracy : {acc:.4f}")
    print(f"  F1 macro : {f1:.4f}")
    print(f"  Train    : {t_train:.1f}s")
    print("\n" + classification_report(y_test, y_pred,
                                       target_names=le.classes_, zero_division=0))

    # ── Save ─────────────────────────────────────────────────────────────────
    meta = {
        'accuracy': acc,
        'f1_macro': f1,
        'train_rows': len(X_train),
        'test_rows': len(X_test),
        'classes': le.classes_.tolist(),
        'tfidf_params': TFIDF_PARAMS,
        'lr_params': LR_PARAMS,
    }
    joblib.dump(tfidf,  TFIDF_PATH)
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(le,     LABEL_PATH)
    joblib.dump(meta,   META_PATH)
    print(f"\n✅ Model đã lưu vào thư mục: {MODEL_DIR}/")
    for p in [TFIDF_PATH, MODEL_PATH, LABEL_PATH, META_PATH]:
        size = os.path.getsize(p) / 1024
        print(f"   {p}  ({size:.0f} KB)")


if __name__ == "__main__":
    main()
