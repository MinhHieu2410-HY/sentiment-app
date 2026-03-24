"""
app.py — Sentiment Analysis Web App
=====================================
Streamlit UI cho phân tích cảm xúc review sản phẩm
Model: Logistic Regression + TF-IDF (DoAn_II___.ipynb)
"""

import os
import io
import time
import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

from utils.preprocessing import preprocess_text

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis — Product Reviews",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR  = "models"
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression.joblib")
LABEL_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")
META_PATH  = os.path.join(MODEL_DIR, "model_meta.joblib")

SENTIMENT_COLORS = {
    "positive": "#2ecc71",
    "neutral":  "#f39c12",
    "negative": "#e74c3c",
}
SENTIMENT_EMOJI = {
    "positive": "😊",
    "neutral":  "😐",
    "negative": "😞",
}
SENTIMENT_VI = {
    "positive": "Tích cực",
    "neutral":  "Trung lập",
    "negative": "Tiêu cực",
}

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
body { font-family: 'Segoe UI', sans-serif; }
.main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Sentiment badge ── */
.badge {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 12px 24px; border-radius: 50px;
    font-size: 1.4rem; font-weight: 700; letter-spacing: 0.5px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    margin: 8px 0;
}
.badge-positive { background: linear-gradient(135deg,#2ecc71,#27ae60); color:#fff; }
.badge-neutral  { background: linear-gradient(135deg,#f39c12,#e67e22); color:#fff; }
.badge-negative { background: linear-gradient(135deg,#e74c3c,#c0392b); color:#fff; }

/* ── Metric cards ── */
.metric-card {
    background: white; border-radius: 12px; padding: 16px 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08); border-left: 5px solid;
    margin-bottom: 12px;
}
.metric-positive { border-left-color: #2ecc71; }
.metric-neutral  { border-left-color: #f39c12; }
.metric-negative { border-left-color: #e74c3c; }

/* ── Info box ── */
.info-box {
    background: #eaf4fb; border-left: 4px solid #3498db;
    border-radius: 8px; padding: 12px 16px; margin: 8px 0;
    font-size: 0.9rem;
}

/* ── Section header ── */
.section-header {
    font-size: 1.1rem; font-weight: 600; color: #2c3e50;
    border-bottom: 2px solid #ecf0f1; padding-bottom: 6px;
    margin: 16px 0 12px 0;
}

/* ── History table ── */
.history-item {
    padding: 10px 14px; border-radius: 8px;
    margin: 4px 0; background: #f8f9fa;
    border-left: 4px solid;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model loader (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Đang tải mô hình…")
def load_model():
    """Load TF-IDF + LR + LabelEncoder. Auto-create demo if not found."""
    if not os.path.exists(MODEL_PATH):
        st.warning("⚠️  Model chưa tồn tại — đang tạo demo model…")
        from create_demo_model import build_demo_model
        build_demo_model()

    tfidf = joblib.load(TFIDF_PATH)
    model = joblib.load(MODEL_PATH)
    le    = joblib.load(LABEL_PATH)
    meta  = joblib.load(META_PATH) if os.path.exists(META_PATH) else {}
    return tfidf, model, le, meta


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────
def predict_single(text: str, tfidf, model, le):
    """Return (label, confidence_dict)."""
    processed = preprocess_text(text)
    if not processed.strip():
        return "neutral", {"positive": 0.33, "neutral": 0.34, "negative": 0.33}
    vec = tfidf.transform([processed])
    proba = model.predict_proba(vec)[0]
    classes = le.classes_
    conf_dict = {cls: float(prob) for cls, prob in zip(classes, proba)}
    label = classes[np.argmax(proba)]
    return label, conf_dict


def predict_batch(texts: list, tfidf, model, le) -> list:
    """Return list of (label, confidence) tuples."""
    processed = [preprocess_text(t) for t in texts]
    # replace empty with placeholder
    processed = [p if p.strip() else "unknown" for p in processed]
    vec   = tfidf.transform(processed)
    proba = model.predict_proba(vec)
    preds = le.inverse_transform(np.argmax(proba, axis=1))
    confs = np.max(proba, axis=1)
    return list(zip(preds, confs))


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────────────
def confidence_bar(conf_dict: dict):
    labels = ["positive", "neutral", "negative"]
    values = [conf_dict.get(l, 0) * 100 for l in labels]
    colors = [SENTIMENT_COLORS[l] for l in labels]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition='outside',
        textfont=dict(size=13, color='#2c3e50'),
    ))
    fig.update_layout(
        margin=dict(t=20, b=20, l=10, r=10),
        height=280,
        yaxis=dict(range=[0, 110], title="Xác suất (%)", gridcolor="#ecf0f1"),
        xaxis=dict(title=""),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI"),
        showlegend=False,
    )
    return fig


def sentiment_pie(counts: dict):
    labels = list(counts.keys())
    values = list(counts.values())
    colors = [SENTIMENT_COLORS.get(l, "#95a5a6") for l in labels]

    fig = go.Figure(go.Pie(
        labels=[f"{SENTIMENT_EMOJI[l]} {SENTIMENT_VI[l]}" for l in labels],
        values=values,
        marker=dict(colors=colors, line=dict(color='white', width=2)),
        textinfo='percent+label',
        textfont=dict(size=13),
        hole=0.4,
    ))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.1),
        font=dict(family="Segoe UI"),
    )
    return fig


def top_words_chart(texts: list, sentiment: str, n: int = 15):
    """Bar chart of top N words for a given sentiment."""
    all_words = " ".join(texts).split()
    freq = Counter(all_words).most_common(n)
    if not freq:
        return None
    words, counts = zip(*freq)
    fig = go.Figure(go.Bar(
        x=list(counts)[::-1],
        y=list(words)[::-1],
        orientation='h',
        marker_color=SENTIMENT_COLORS[sentiment],
        marker_line_color='rgba(0,0,0,0.3)',
        marker_line_width=0.5,
    ))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=20),
        height=400,
        xaxis_title="Tần suất",
        yaxis_title="",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI"),
    )
    return fig


def history_trend_chart(history: list):
    """Line chart of cumulative sentiment counts over session."""
    if len(history) < 2:
        return None
    rows = []
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for i, (_, label, _, ts) in enumerate(history, 1):
        if label in counts:
            counts[label] += 1
        for s, c in counts.items():
            rows.append({"Index": i, "Sentiment": SENTIMENT_VI[s], "Count": c})
    df = pd.DataFrame(rows)
    color_map = {
        SENTIMENT_VI["positive"]: SENTIMENT_COLORS["positive"],
        SENTIMENT_VI["neutral"]:  SENTIMENT_COLORS["neutral"],
        SENTIMENT_VI["negative"]: SENTIMENT_COLORS["negative"],
    }
    fig = px.line(df, x="Index", y="Count", color="Sentiment",
                  color_discrete_map=color_map,
                  markers=True, line_shape="spline")
    fig.update_layout(
        margin=dict(t=10, b=10),
        height=260,
        xaxis_title="Review #",
        yaxis_title="Số lượng tích lũy",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.2),
        font=dict(family="Segoe UI"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []  # [(text, label, conf_dict, timestamp), …]

if "batch_results" not in st.session_state:
    st.session_state.batch_results = None


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Sentiment Analyzer")
    st.markdown("---")

    page = st.radio(
        "📌 Chọn trang",
        ["🔍 Phân tích đơn lẻ", "📂 Phân tích hàng loạt", "📊 Dashboard"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Model info
    tfidf, model, le, meta = load_model()
    is_demo = meta.get("is_demo", False)

    st.markdown("### 📈 Phiên này")
    n_pos = sum(1 for _, l, _, _ in st.session_state.history if l == "positive")
    n_neu = sum(1 for _, l, _, _ in st.session_state.history if l == "neutral")
    n_neg = sum(1 for _, l, _, _ in st.session_state.history if l == "negative")
    total = len(st.session_state.history)

    st.markdown(f"😊 Positive: **{n_pos}**")
    st.markdown(f"😐 Neutral: **{n_neu}**")
    st.markdown(f"😞 Negative: **{n_neg}**")
    st.markdown(f"📝 Tổng: **{total}**")

    if total > 0:
        st.button(
            "🗑️ Xóa lịch sử",
            use_container_width=True,
            on_click=clear_history)
# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Single Review Analysis
# ─────────────────────────────────────────────────────────────────────────────
if "Phân tích đơn lẻ" in page:
    st.markdown("# 🔍 Phân tích Cảm xúc Review")
    st.markdown("Nhập một đoạn review bằng **tiếng Anh** để phân tích cảm xúc.")

    # ── Input
    review_text = st.text_area(
        "✏️ Nhập review của bạn:",
        value="",
        height=140,
        placeholder="e.g. This product is absolutely amazing, I love it so much!",
        key="review_input",                     # this key is what we will reset
    )
    
    col_btn, col_clear = st.columns([1, 4])
    analyze_btn = col_btn.button("🚀 Phân tích", type="primary", use_container_width=True)
    
    if col_clear.button("🔄 Xóa", use_container_width=False):
        st.session_state["review_input"] = ""  
        st.rerun()

    word_count = len(review_text.split()) if review_text.strip() else 0
    st.caption(f"📝 {word_count} từ")

    if analyze_btn:
        if not review_text.strip():
            st.warning("⚠️ Vui lòng nhập nội dung review!")
        else:
            with st.spinner("🔄 Đang phân tích…"):
                t0 = time.time()
                label, conf = predict_single(review_text, tfidf, model, le)
                elapsed = time.time() - t0

            # Save to history
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            st.session_state.history.append((review_text, label, conf, ts))

            # ── Results layout
            st.markdown("---")
            st.markdown("## 📊 Kết quả phân tích")

            res_col1, res_col2 = st.columns([1, 1.4])

            with res_col1:
                # Badge
                badge_cls = f"badge badge-{label}"
                emoji = SENTIMENT_EMOJI[label]
                label_vi = SENTIMENT_VI[label]
                conf_pct = conf[label] * 100
                st.markdown(
                    f'<div class="{badge_cls}">{emoji} {label_vi.upper()}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Độ tin cậy:** `{conf_pct:.1f}%`")
                st.markdown(f"⏱️ Thời gian: `{elapsed*1000:.1f} ms`")

                # Confidence table
                st.markdown('<p class="section-header">📋 Xác suất từng nhãn</p>',
                            unsafe_allow_html=True)
                for sent in ["positive", "neutral", "negative"]:
                    pct = conf.get(sent, 0) * 100
                    is_max = sent == label
                    icon = "✅" if is_max else "  "
                    bar_width = int(pct)
                    color = SENTIMENT_COLORS[sent]
                    st.markdown(
                        f"""
                        <div style="margin:6px 0">
                          <div style="display:flex;justify-content:space-between;margin-bottom:3px">
                            <span>{icon} {SENTIMENT_EMOJI[sent]} {SENTIMENT_VI[sent]}</span>
                            <b>{pct:.1f}%</b>
                          </div>
                          <div style="background:#ecf0f1;border-radius:6px;height:10px">
                            <div style="width:{bar_width}%;background:{color};
                                        height:10px;border-radius:6px;transition:0.3s"></div>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with res_col2:
                st.markdown('<p class="section-header">📈 Biểu đồ độ tin cậy</p>',
                            unsafe_allow_html=True)
                st.plotly_chart(confidence_bar(conf), use_container_width=True,
                                config={"displayModeBar": False})

            # ── Processed text
            with st.expander("🔬 Văn bản sau tiền xử lý"):
                processed = preprocess_text(review_text)
                st.code(processed if processed else "(empty after processing)", language=None)
                st.caption("→ Lowercase → remove URLs/punctuation → stopwords → lemmatize")

            # ── Quick insight
            st.markdown("---")
            if label == "positive":
                st.success(f"✅ Review này **tích cực**! Khách hàng hài lòng với sản phẩm.")
            elif label == "negative":
                st.error(f"❌ Review này **tiêu cực**. Cần chú ý phản hồi của khách hàng.")
            else:
                st.info(f"ℹ️ Review này **trung lập**. Khách hàng có ý kiến trung dung.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — Batch Analysis
# ─────────────────────────────────────────────────────────────────────────────
elif "Phân tích hàng loạt" in page:
    st.markdown("# 📂 Phân tích Hàng Loạt")
    st.markdown("Upload file CSV chứa nhiều reviews để phân tích đồng loạt.")

    st.markdown(
        '<div class="info-box">📌 <b>Yêu cầu file CSV:</b> Phải có cột chứa nội dung review '
        '(tên cột thường là <code>review</code>, <code>text</code>, <code>comment</code>…). '
        'Mỗi dòng = một review.</div>',
        unsafe_allow_html=True,
    )

    # ── Demo CSV download
    demo_df = pd.DataFrame({
        "review": [
            "This product is amazing, I absolutely love it!",
            "Terrible quality, broke after one day, waste of money.",
            "It is okay, nothing special but does the job.",
            "Excellent performance, highly recommend to everyone.",
            "Very disappointed, does not work as advertised.",
            "Average product, meets basic requirements.",
            "Fantastic purchase, great value for money!",
            "Worst product ever, completely useless garbage.",
        ]
    })
    demo_csv = demo_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Tải file CSV mẫu",
        data=demo_csv,
        file_name="sample_reviews.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("📎 Upload file CSV", type=["csv"])

    if uploaded:
        try:
            df_up = pd.read_csv(uploaded, encoding='utf-8')
        except UnicodeDecodeError:
            df_up = pd.read_csv(uploaded, encoding='latin1')

        st.success(f"✅ Đã tải: **{len(df_up):,} dòng** × {df_up.shape[1]} cột")
        st.dataframe(df_up.head(5), use_container_width=True)

        # ── Column selector
        text_cols = [c for c in df_up.columns
                     if any(k in c.lower() for k in
                            ['review','text','comment','content','feedback','description'])]
        if not text_cols:
            text_cols = list(df_up.columns)

        selected_col = st.selectbox(
            "🔤 Chọn cột chứa nội dung review:",
            options=list(df_up.columns),
            index=list(df_up.columns).index(text_cols[0]) if text_cols else 0,
        )

        max_rows = st.slider(
            "⚙️ Số dòng tối đa xử lý:", 10, min(10_000, len(df_up)),
            min(1_000, len(df_up)), step=100,
        )

        if st.button("🚀 Bắt đầu phân tích", type="primary"):
            df_proc = df_up[[selected_col]].dropna().head(max_rows).copy()
            df_proc.columns = ["review"]

            progress = st.progress(0, "Đang xử lý…")
            texts = df_proc["review"].tolist()

            # Process in chunks
            chunk = 500
            results = []
            for i in range(0, len(texts), chunk):
                batch = texts[i:i+chunk]
                results.extend(predict_batch(batch, tfidf, model, le))
                progress.progress(min((i + chunk) / len(texts), 1.0))

            df_proc["sentiment"]   = [r[0] for r in results]
            df_proc["confidence"]  = [round(r[1] * 100, 1) for r in results]
            df_proc["emoji"]       = df_proc["sentiment"].map(SENTIMENT_EMOJI)
            df_proc["label_vi"]    = df_proc["sentiment"].map(SENTIMENT_VI)
            progress.empty()

            st.session_state.batch_results = df_proc
            st.success(f"✅ Đã phân tích **{len(df_proc):,}** reviews!")

    # ── Show batch results
    if st.session_state.batch_results is not None:
        df_r = st.session_state.batch_results
        st.markdown("---")
        st.markdown("## 📊 Kết quả")

        # Metrics row
        counts = df_r["sentiment"].value_counts().to_dict()
        total  = len(df_r)
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("📝 Tổng", f"{total:,}")
        mc2.metric(f"😊 Positive", f"{counts.get('positive',0):,}",
                   f"{counts.get('positive',0)/total*100:.1f}%")
        mc3.metric(f"😐 Neutral",  f"{counts.get('neutral',0):,}",
                   f"{counts.get('neutral',0)/total*100:.1f}%")
        mc4.metric(f"😞 Negative", f"{counts.get('negative',0):,}",
                   f"{counts.get('negative',0)/total*100:.1f}%")

        # Charts
        ch1, ch2 = st.columns(2)
        with ch1:
            st.markdown("#### 🥧 Phân phối Sentiment")
            st.plotly_chart(sentiment_pie(counts), use_container_width=True,
                            config={"displayModeBar": False})
        with ch2:
            st.markdown("#### 📊 Phân phối Confidence")
            fig_hist = px.histogram(
                df_r, x="confidence", color="sentiment",
                color_discrete_map=SENTIMENT_COLORS,
                nbins=20, barmode='overlay', opacity=0.75,
                labels={"confidence": "Confidence (%)", "sentiment": ""},
            )
            fig_hist.update_layout(
                margin=dict(t=10, b=10), height=300,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=-0.2),
                font=dict(family="Segoe UI"),
            )
            st.plotly_chart(fig_hist, use_container_width=True,
                            config={"displayModeBar": False})

        # Filter table
        st.markdown("#### 📋 Bảng kết quả")
        filter_sent = st.multiselect(
            "Lọc theo Sentiment:",
            ["positive", "neutral", "negative"],
            default=["positive", "neutral", "negative"],
        )
        df_show = df_r[df_r["sentiment"].isin(filter_sent)] if filter_sent else df_r
        st.dataframe(
            df_show[["emoji", "review", "label_vi", "confidence"]]
              .rename(columns={"emoji": "", "review": "Review",
                               "label_vi": "Sentiment", "confidence": "Confidence (%)"}),
            use_container_width=True, height=380,
        )

        # Export
        st.markdown("#### 💾 Xuất kết quả")
        csv_out = df_r[["review", "sentiment", "confidence"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Tải kết quả (.csv)",
            data=csv_out,
            file_name=f"sentiment_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Dashboard
# ─────────────────────────────────────────────────────────────────────────────
elif "Dashboard" in page:
    st.markdown("# 📊 Dashboard Tổng Hợp")

    history = st.session_state.history
    batch   = st.session_state.batch_results

    # ── Combine sources
    has_history = len(history) > 0
    has_batch   = batch is not None and len(batch) > 0

    if not has_history and not has_batch:
        st.info("ℹ️ Chưa có dữ liệu. Hãy phân tích một số reviews ở trang **Phân tích đơn lẻ** "
                "hoặc upload file ở trang **Phân tích hàng loạt**.")
        st.stop()

    # Build combined df
    rows = []
    if has_history:
        for text, label, conf, ts in history:
            rows.append({"review": text, "sentiment": label,
                         "confidence": round(conf[label]*100,1), "source": "Single"})
    if has_batch:
        for _, row in batch.iterrows():
            rows.append({"review": row["review"], "sentiment": row["sentiment"],
                         "confidence": row["confidence"], "source": "Batch"})
    df_all = pd.DataFrame(rows)

    # ── Top KPIs
    st.markdown("### 📌 Tổng quan")
    counts = df_all["sentiment"].value_counts().to_dict()
    total  = len(df_all)
    avg_conf = df_all["confidence"].mean()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("📝 Tổng reviews",  f"{total:,}")
    k2.metric("😊 Positive",  f"{counts.get('positive',0):,}",
              f"{counts.get('positive',0)/total*100:.1f}%")
    k3.metric("😐 Neutral",   f"{counts.get('neutral',0):,}",
              f"{counts.get('neutral',0)/total*100:.1f}%")
    k4.metric("😞 Negative",  f"{counts.get('negative',0):,}",
              f"{counts.get('negative',0)/total*100:.1f}%")
    k5.metric("🎯 Avg Confidence", f"{avg_conf:.1f}%")

    st.markdown("---")

    # ── Charts row 1
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🥧 Phân phối tổng thể")
        st.plotly_chart(sentiment_pie(counts), use_container_width=True,
                        config={"displayModeBar": False})
    with c2:
        st.markdown("#### 📊 Confidence theo nhãn")
        fig_box = px.box(
            df_all, x="sentiment", y="confidence", color="sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            labels={"sentiment": "", "confidence": "Confidence (%)"},
        )
        fig_box.update_layout(
            margin=dict(t=10, b=10), height=320,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, font=dict(family="Segoe UI"),
        )
        st.plotly_chart(fig_box, use_container_width=True,
                        config={"displayModeBar": False})

    # ── Trend (session history only)
    if has_history and len(history) >= 2:
        st.markdown("#### 📈 Xu hướng phân tích trong phiên")
        fig_trend = history_trend_chart(history)
        if fig_trend:
            st.plotly_chart(fig_trend, use_container_width=True,
                            config={"displayModeBar": False})

    # ── Top words per sentiment
    st.markdown("---")
    st.markdown("#### 🔤 Từ khóa nổi bật theo nhãn")
    tw1, tw2, tw3 = st.columns(3)
    for col, sent in zip([tw1, tw2, tw3], ["positive", "neutral", "negative"]):
        df_sent = df_all[df_all["sentiment"] == sent]
        col.markdown(f"**{SENTIMENT_EMOJI[sent]} {SENTIMENT_VI[sent]}** ({len(df_sent):,})")
        if len(df_sent) > 0:
            all_texts = " ".join(df_sent["review"].apply(preprocess_text).tolist())
            words = [w for w in all_texts.split() if len(w) > 2]
            freq = Counter(words).most_common(10)
            if freq:
                words_list, counts_list = zip(*freq)
                fig_tw = go.Figure(go.Bar(
                    x=list(counts_list)[::-1], y=list(words_list)[::-1],
                    orientation='h', marker_color=SENTIMENT_COLORS[sent],
                ))
                fig_tw.update_layout(
                    margin=dict(t=5, b=5, l=10, r=10), height=280,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Segoe UI", size=11),
                )
                col.plotly_chart(fig_tw, use_container_width=True,
                                 config={"displayModeBar": False})
        else:
            col.caption("Chưa có dữ liệu")

    # ── Source breakdown
    if has_history and has_batch:
        st.markdown("---")
        st.markdown("#### 📂 Theo nguồn dữ liệu")
        fig_src = px.histogram(
            df_all, x="sentiment", color="source", barmode="group",
            color_discrete_sequence=["#3498db", "#9b59b6"],
            labels={"sentiment": "", "source": "Nguồn"},
        )
        fig_src.update_layout(
            margin=dict(t=10, b=10), height=280,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Segoe UI"),
        )
        st.plotly_chart(fig_src, use_container_width=True,
                        config={"displayModeBar": False})

    # ── Export full report
    st.markdown("---")
    st.markdown("#### 💾 Xuất báo cáo")
    report_csv = df_all[["review", "sentiment", "confidence", "source"]].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Tải toàn bộ dữ liệu (.csv)",
        data=report_csv,
        file_name=f"sentiment_dashboard_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )
