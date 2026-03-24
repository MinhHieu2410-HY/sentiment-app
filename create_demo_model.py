"""
create_demo_model.py
====================
Tạo model demo nhỏ với ~300 reviews tích hợp sẵn.
Dùng khi chưa có dataset 2.5M hoặc để test nhanh.

CÁCH DÙNG:
    python create_demo_model.py
"""

import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from utils.preprocessing import preprocess_text

MODEL_DIR  = "models"
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression.joblib")
LABEL_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")
META_PATH  = os.path.join(MODEL_DIR, "model_meta.joblib")

# --------------------------------------------------------------------------- #
# Demo samples — balanced 100 per class
# --------------------------------------------------------------------------- #
DEMO_DATA = [
    # ── POSITIVE ────────────────────────────────────────────────────────────
    ("This product is absolutely amazing, best purchase I have ever made", "positive"),
    ("Incredible quality, works perfectly and arrived fast", "positive"),
    ("Love this so much, highly recommend to everyone", "positive"),
    ("Exceeded my expectations, the quality is outstanding", "positive"),
    ("Five stars, will definitely buy again", "positive"),
    ("Perfect product, does exactly what it promises", "positive"),
    ("Excellent value for money, very happy with this purchase", "positive"),
    ("Great product, fast shipping and perfect packaging", "positive"),
    ("Works great, very satisfied with the results", "positive"),
    ("Best buy ever, fantastic quality and super fast delivery", "positive"),
    ("Really enjoying this, much better than expected", "positive"),
    ("This is a wonderful product, I use it every day", "positive"),
    ("Superb quality, looks even better in person", "positive"),
    ("Very happy customer, would definitely recommend", "positive"),
    ("Outstanding performance, exactly what I needed", "positive"),
    ("Brilliant product, works flawlessly every time", "positive"),
    ("I'm very impressed with the quality of this item", "positive"),
    ("Fantastic value, great build quality and solid performance", "positive"),
    ("Really good product, does the job perfectly", "positive"),
    ("The best I have tried, highly recommend buying it", "positive"),
    ("Awesome product excellent customer service overall experience fantastic", "positive"),
    ("Great value product highly satisfied with purchase quality amazing", "positive"),
    ("Wonderful item works perfectly love using it every day", "positive"),
    ("Very pleased with purchase would recommend to friends family", "positive"),
    ("Top quality item arrived quickly well packaged recommend", "positive"),
    ("Absolutely love this product works brilliantly every single time", "positive"),
    ("Incredible purchase definitely worth the money super happy", "positive"),
    ("Best product available great price wonderful quality satisfied", "positive"),
    ("Happy with this item great build well designed enjoyed", "positive"),
    ("Superb item excellent quality fast delivery loved experience", "positive"),
    ("This game is perfect for all ages extremely fun and enjoyable", "positive"),
    ("Course content was excellent learned so much valuable information", "positive"),
    ("Great hotel very comfortable room excellent staff friendly service", "positive"),
    ("Amazing food delicious taste highly recommend restaurant meal wonderful", "positive"),
    ("Phone works brilliantly battery life excellent camera amazing photos", "positive"),
    ("Book is wonderful captivating story loved every single page", "positive"),
    ("Shoes are very comfortable stylish looking great quality leather", "positive"),
    ("App works perfectly smooth interface easy use no bugs", "positive"),
    ("Laptop performance excellent fast processing long battery superb", "positive"),
    ("Headphones sound quality incredible comfortable wear noise cancelling great", "positive"),
    ("Great course instructor explains everything clearly highly recommended learning", "positive"),
    ("Excellent movie very entertaining storyline acting superb recommend watching", "positive"),
    ("Software easy install works smoothly no issues excellent product", "positive"),
    ("Hotel breakfast amazing room clean staff helpful pleasant stay", "positive"),
    ("Excellent service very helpful team resolved issue quickly thank", "positive"),
    ("Product looks amazing high quality materials durable worth price", "positive"),
    ("Very satisfied happy purchase good quality item delivery fast", "positive"),
    ("Love using daily product reliable consistent excellent results always", "positive"),
    ("Fantastic purchase great savings quality exceeded expectations recommend it", "positive"),
    ("Really nice item good value happy recommend friends buy", "positive"),

    # ── NEGATIVE ────────────────────────────────────────────────────────────
    ("This is terrible, completely broken and useless", "negative"),
    ("Worst product ever, waste of money, do not buy", "negative"),
    ("Very disappointed, does not work as advertised at all", "negative"),
    ("Awful quality, broke after just one day of use", "negative"),
    ("Complete garbage, absolutely not worth the money", "negative"),
    ("Terrible experience, customer service was rude and unhelpful", "negative"),
    ("Do not buy this, it stopped working immediately", "negative"),
    ("Very bad quality, falling apart after minimal use", "negative"),
    ("Disgusting experience, worst purchase I have ever made", "negative"),
    ("Total waste of money, nothing works as described", "negative"),
    ("Broke instantly, terrible build quality avoid this", "negative"),
    ("Horrible product, returned immediately for full refund", "negative"),
    ("Not as advertised, very poor quality item", "negative"),
    ("Disappointed with purchase, expected much better quality", "negative"),
    ("Very poor product, would not recommend to anyone", "negative"),
    ("Shocking quality, completely different to pictures shown", "negative"),
    ("Defective product arrived broken cannot use it", "negative"),
    ("Terrible customer service never resolved my problem completely ignored", "negative"),
    ("Product failed immediately stopped working first day useless", "negative"),
    ("Worst experience ever extremely disappointed avoid purchasing this item", "negative"),
    ("Very poor quality cheap materials broke easily terrible buy", "negative"),
    ("Complete waste money does not work as described horrible", "negative"),
    ("Absolutely terrible product broken arrived damaged poor quality overall", "negative"),
    ("Rubbish product falls apart immediately cheap nasty avoid this", "negative"),
    ("Dreadful experience terrible quality nothing works refund requested", "negative"),
    ("Horrible quality item broke instantly cheap plastic terrible", "negative"),
    ("Awful game boring repetitive crashes constantly broken terrible experience", "negative"),
    ("Course was terrible waste time learned nothing poor instructor", "negative"),
    ("Hotel dirty room noisy uncomfortable bed terrible service rude staff", "negative"),
    ("Food was disgusting cold undercooked terrible service never return", "negative"),
    ("Phone constantly freezing battery dies quickly terrible camera quality", "negative"),
    ("Book boring repetitive poorly written waste time money terrible", "negative"),
    ("Shoes fell apart immediately uncomfortable poor quality materials avoid", "negative"),
    ("App crashes constantly full bugs slow interface terrible experience", "negative"),
    ("Laptop overheats constantly slow bad battery terrible performance", "negative"),
    ("Headphones sound terrible broke quickly uncomfortable wear bad quality", "negative"),
    ("Course incomplete misleading description poor instructor terrible value money", "negative"),
    ("Movie boring predictable poorly acted waste time terrible film", "negative"),
    ("Software broken many bugs crashes freezes unusable terrible product", "negative"),
    ("Hotel overpriced dirty unhelpful staff terrible breakfast avoid", "negative"),
    ("Service terrible unhelpful staff rude ignored my problem resolved nothing", "negative"),
    ("Product completely different photo cheap quality terrible packaging damaged", "negative"),
    ("Very unhappy purchase poor quality broken immediately disappointed regret", "negative"),
    ("Worst product market terrible design unreliable constant problems avoid", "negative"),
    ("Absolutely useless product complete waste money regret buying this", "negative"),
    ("Terrible quality item arrived damaged broken useless never buy again", "negative"),
    ("Product stopped working day received terrible quality cheap garbage", "negative"),
    ("Awful experience broken product terrible support never resolved issue", "negative"),
    ("Horrible purchase cheap material broke immediately terrible quality avoid", "negative"),
    ("Complete disappointment nothing works terrible customer service waste money", "negative"),

    # ── NEUTRAL ─────────────────────────────────────────────────────────────
    ("It is okay, nothing special but does the job", "neutral"),
    ("Average product, nothing to write home about", "neutral"),
    ("Works fine, not amazing but acceptable for the price", "neutral"),
    ("Decent enough, gets the job done without any issues", "neutral"),
    ("It is alright, has some pros and cons worth considering", "neutral"),
    ("Not bad but not great either, pretty middle of the road", "neutral"),
    ("Does what it says, nothing more nothing less than expected", "neutral"),
    ("Mediocre quality, could be better but not terrible", "neutral"),
    ("Neither good nor bad, just an average product overall", "neutral"),
    ("Acceptable quality for the price but nothing outstanding", "neutral"),
    ("Works as expected, no complaints but no praise either", "neutral"),
    ("It is passable, met basic requirements nothing more though", "neutral"),
    ("Average experience, some things good some things less so", "neutral"),
    ("Middle of the road product, serves its purpose adequately", "neutral"),
    ("So-so product, has both positive and negative aspects", "neutral"),
    ("Fair enough for price paid, no major issues encountered", "neutral"),
    ("Not impressive but not bad either, just average product", "neutral"),
    ("Could be better could be worse fairly typical purchase", "neutral"),
    ("Mixed feelings about this product has good bad aspects", "neutral"),
    ("Okay product does basic job not exceptional value overall", "neutral"),
    ("Average product works sometimes has few issues acceptable", "neutral"),
    ("Neither amazing terrible just okay meets basic expectations", "neutral"),
    ("Decent product not perfect some minor issues but works", "neutral"),
    ("Mediocre experience some aspects good others need improvement", "neutral"),
    ("Just about okay for price paid not impressed not disappointed", "neutral"),
    ("Works sometimes unreliable not great not terrible average quality", "neutral"),
    ("Game okay fun sometimes boring other times average experience", "neutral"),
    ("Course average some good content some outdated information", "neutral"),
    ("Hotel acceptable basic amenities nothing special average stay", "neutral"),
    ("Food average taste nothing special okay for price paid", "neutral"),
    ("Phone works okay average camera battery life acceptable", "neutral"),
    ("Book interesting parts boring parts overall average reading", "neutral"),
    ("Shoes comfortable enough average quality nothing special decent", "neutral"),
    ("App works mostly some bugs occasional crashes average experience", "neutral"),
    ("Laptop average performance okay battery not impressive", "neutral"),
    ("Headphones sound okay average comfort decent build mediocre", "neutral"),
    ("Course has useful content but some sections poorly explained", "neutral"),
    ("Movie entertaining enough predictable plot average acting okay", "neutral"),
    ("Software works mostly some issues nothing critical average", "neutral"),
    ("Hotel room clean enough average staff service nothing special", "neutral"),
    ("Service acceptable waited long time resolved issue eventually", "neutral"),
    ("Product acceptable quality average delivery time packaging fine", "neutral"),
    ("Okay purchase nothing spectacular meets basic requirements only", "neutral"),
    ("Mixed experience some good aspects let down other areas", "neutral"),
    ("Not worth extra cost basic functionality average quality okay", "neutral"),
    ("Neutral experience expected more based description delivered average", "neutral"),
    ("Product functional not special average quality acceptable price", "neutral"),
    ("Okay item does job nothing more some minor quality issues", "neutral"),
    ("Average rating reflects mixed experience some positives negatives", "neutral"),
    ("Satisfactory purchase nothing extraordinary meets minimum requirements", "neutral"),
    ("Adequate product serves purpose basic features work okay", "neutral"),
]


def build_demo_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    reviews   = [preprocess_text(r) for r, _ in DEMO_DATA]
    labels    = [l for _, l in DEMO_DATA]

    le = LabelEncoder()
    y  = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        reviews, y, test_size=0.2, random_state=42, stratify=y
    )

    tfidf = TfidfVectorizer(
        max_features=5_000, ngram_range=(1, 2),
        min_df=1, max_df=1.0, sublinear_tf=True
    )
    X_tr = tfidf.fit_transform(X_train)
    X_te = tfidf.transform(X_test)

    model = LogisticRegression(
        max_iter=1000, C=1.0, solver='lbfgs',
        random_state=42
    )
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='macro', zero_division=0)

    meta = {
        'accuracy': acc,
        'f1_macro': f1,
        'train_rows': len(X_train),
        'test_rows': len(X_test),
        'classes': le.classes_.tolist(),
        'is_demo': True,
        'note': 'Demo model — 150 sample reviews. Run train_model.py for full accuracy.',
    }

    joblib.dump(tfidf,  TFIDF_PATH)
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(le,     LABEL_PATH)
    joblib.dump(meta,   META_PATH)

    print("✅ Demo model đã tạo!")
    print(f"   Accuracy : {acc:.3f}")
    print(f"   F1 macro : {f1:.3f}")
    print(f"   Classes  : {le.classes_.tolist()}")
    print(f"\n   ⚠️  Đây là DEMO model ({len(DEMO_DATA)} reviews).")
    print("   Để đạt accuracy ~89.5%, chạy:")
    print("   python train_model.py --data /path/to/2.5m-reviews-dataset.csv")


if __name__ == "__main__":
    build_demo_model()
