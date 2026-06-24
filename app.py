import os
import re
import glob
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

try:
    import pycountry
except Exception:
    pycountry = None

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="YouTube Views Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(249,115,22,0.20), transparent 30%),
        radial-gradient(circle at top right, rgba(34,197,94,0.18), transparent 32%),
        linear-gradient(135deg, #0b1120 0%, #111827 55%, #0f172a 100%);
    color: #e5e7eb;
}

[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.94);
    border-right: 1px solid rgba(148, 163, 184, 0.18);
}

[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

.main-title {
    font-size: 44px;
    font-weight: 900;
    line-height: 1.1;
    margin-bottom: 8px;
    background: linear-gradient(90deg, #fb923c, #22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    font-size: 16px;
    color: #cbd5e1;
    margin-bottom: 26px;
}

.metric-box {
    background: linear-gradient(145deg, rgba(15,23,42,0.95), rgba(30,41,59,0.82));
    border: 1px solid rgba(148,163,184,0.22);
    border-radius: 20px;
    padding: 20px;
    min-height: 120px;
    box-shadow: 0 16px 38px rgba(0,0,0,0.24);
}

.metric-label {
    color: #94a3b8;
    font-size: 13px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

.metric-value {
    color: #f8fafc;
    font-size: 30px;
    font-weight: 900;
    margin-top: 8px;
}

.metric-desc {
    color: #cbd5e1;
    font-size: 13px;
    margin-top: 8px;
}

.card {
    background: rgba(15, 23, 42, 0.82);
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 20px;
    padding: 22px;
    box-shadow: 0 16px 40px rgba(0,0,0,0.28);
}

.section-title {
    font-size: 24px;
    font-weight: 900;
    color: #f8fafc;
    margin-bottom: 6px;
}

.section-desc {
    color: #cbd5e1;
    margin-bottom: 18px;
}

.prediction-box {
    background: linear-gradient(135deg, #f97316, #22c55e);
    color: white;
    padding: 32px;
    border-radius: 24px;
    text-align: center;
    box-shadow: 0 18px 50px rgba(0,0,0,0.35);
}

.prediction-label {
    font-size: 16px;
    font-weight: 700;
    opacity: 0.94;
}

.prediction-value {
    font-size: 46px;
    font-weight: 900;
    margin-top: 8px;
}

.tier-box {
    border-radius: 18px;
    padding: 16px;
    color: white;
    margin-top: 12px;
    text-align: center;
    font-weight: 800;
}

.tier-under {
    background: linear-gradient(135deg, #3b82f6, #1e40af);
}

.tier-bronze {
    background: linear-gradient(135deg, #22c55e, #166534);
}

.tier-silver {
    background: linear-gradient(135deg, #facc15, #ca8a04);
}

.tier-gold {
    background: linear-gradient(135deg, #fb923c, #c2410c);
}

.tier-diamond {
    background: linear-gradient(135deg, #ef4444, #991b1b);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    background-color: rgba(15,23,42,0.88);
    border-radius: 14px;
    color: #cbd5e1;
    padding: 12px 18px;
    border: 1px solid rgba(148,163,184,0.20);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #f97316, #22c55e);
    color: white;
}

hr {
    border: none;
    height: 1px;
    background: rgba(148, 163, 184, 0.18);
    margin: 24px 0;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# COUNTRY HELPER
# ============================================================

COUNTRY_OVERRIDES = {
    "UK": "United Kingdom",
    "UNKNOWN": "Unknown"
}


def normalize_country_code(value):
    value = str(value).strip()

    if value == "" or value.lower() in ["nan", "none", "null", "unknown"]:
        return "Unknown"

    if len(value) == 2:
        return value.upper()

    return value


def country_label(value):
    value = normalize_country_code(value)
    upper = str(value).upper()

    if upper in ["", "NAN", "NONE", "NULL", "UNKNOWN"]:
        return "Unknown"

    if upper in COUNTRY_OVERRIDES:
        return f"{upper} - {COUNTRY_OVERRIDES[upper]}"

    if pycountry is not None:
        try:
            country = pycountry.countries.get(alpha_2=upper)
            if country:
                return f"{upper} - {country.name}"
        except Exception:
            pass

    return value


def feature_label(feature_name):
    if feature_name == "subscribers":
        return "Subscribers"

    if feature_name == "total_videos":
        return "Total Videos"

    if feature_name.startswith("country_"):
        code = feature_name.replace("country_", "")
        return "Country: " + country_label(code)

    return feature_name


# ============================================================
# GENERAL HELPER
# ============================================================

def format_number(value):
    try:
        value = float(value)
    except Exception:
        return "-"

    if value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.2f}T"
    elif value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.2f}K"
    else:
        return f"{value:,.0f}"


def format_full_number(value):
    try:
        return f"{float(value):,.0f}"
    except Exception:
        return "-"


def parse_number(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).strip().lower()
    text = text.replace(",", "").replace(" ", "")

    multiplier = 1

    if text.endswith("k"):
        multiplier = 1_000
        text = text[:-1]
    elif text.endswith("m"):
        multiplier = 1_000_000
        text = text[:-1]
    elif text.endswith("b"):
        multiplier = 1_000_000_000
        text = text[:-1]
    elif text.endswith("t"):
        multiplier = 1_000_000_000_000
        text = text[:-1]

    text = re.sub(r"[^0-9.\-]", "", text)

    try:
        return float(text) * multiplier
    except Exception:
        return np.nan


def normalize_col(col):
    return str(col).lower().replace("_", " ").replace("-", " ").strip()


def find_col(df, candidates):
    normalized = {normalize_col(col): col for col in df.columns}

    for candidate in candidates:
        candidate_norm = normalize_col(candidate)
        if candidate_norm in normalized:
            return normalized[candidate_norm]

    for col_norm, original_col in normalized.items():
        for candidate in candidates:
            candidate_norm = normalize_col(candidate)
            if candidate_norm in col_norm:
                return original_col

    return None


def get_tier_info(subscribers):
    if subscribers < 1_000_000:
        return "🔵 <1M", "tier-under", "Channel berada di bawah 1 juta subscribers."
    elif subscribers < 10_000_000:
        return "🟢 1–10M", "tier-bronze", "Channel berada pada rentang 1 juta sampai kurang dari 10 juta subscribers."
    elif subscribers < 50_000_000:
        return "🟡 10–50M", "tier-silver", "Channel berada pada rentang 10 juta sampai kurang dari 50 juta subscribers."
    elif subscribers < 100_000_000:
        return "🟠 50–100M", "tier-gold", "Channel berada pada rentang 50 juta sampai kurang dari 100 juta subscribers."
    else:
        return "🔴 100M+", "tier-diamond", "Channel berada pada 100 juta subscribers atau lebih."


def get_tier(subscribers):
    return get_tier_info(subscribers)[0]


def metric_card(label, value, desc=""):
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)


def find_dataset_file():
    possible_paths = [
        "youtube_4k_channels_cleaned.csv",
        "/content/youtube_4k_channels_cleaned.csv",
        "/mnt/data/youtube_4k_channels_cleaned.csv",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    for pattern in ["*.csv", "/content/*.csv", "/mnt/data/*.csv"]:
        for path in glob.glob(pattern):
            name = os.path.basename(path).lower()
            if "youtube" in name or "channel" in name:
                return path

    return None


@st.cache_data
def load_csv(file_or_path):
    return pd.read_csv(file_or_path)


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data(df_raw):
    df = df_raw.copy()

    sub_col = find_col(df, [
        "Subscribers",
        "Subscriber",
        "Subscriber Count",
        "Subscribers Count",
        "Subscribers (M)",
    ])

    views_col = find_col(df, [
        "Total Views",
        "Views",
        "View Count",
        "Video Views",
        "Total Video Views",
        "Views (B)",
    ])

    videos_col = find_col(df, [
        "Total Videos",
        "Videos",
        "Video Count",
        "Uploads",
        "Total Uploads",
    ])

    country_col = find_col(df, [
        "Country",
        "Country Code",
        "Abbreviation",
        "Location",
        "Channel Country",
    ])

    channel_col = find_col(df, [
        "Channel Name",
        "Youtuber",
        "Name",
        "Title",
        "Channel",
    ])

    rank_col = find_col(df, [
        "Global Rank",
        "Rank",
    ])

    description_col = find_col(df, [
        "Description",
        "Desc",
    ])

    missing = []

    if sub_col is None:
        missing.append("Subscribers")

    if views_col is None:
        missing.append("Total Views")

    if videos_col is None:
        missing.append("Total Videos")

    if missing:
        st.error("Kolom wajib tidak ditemukan: " + ", ".join(missing))
        st.write("Kolom yang tersedia pada dataset:")
        st.write(list(df.columns))
        st.stop()

    data = pd.DataFrame()

    data["subscribers"] = df[sub_col].apply(parse_number)
    data["total_views"] = df[views_col].apply(parse_number)
    data["total_videos"] = df[videos_col].apply(parse_number)

    if "(m)" in normalize_col(sub_col) and data["subscribers"].median() < 10000:
        data["subscribers"] = data["subscribers"] * 1_000_000

    if "(b)" in normalize_col(views_col) and data["total_views"].median() < 10000:
        data["total_views"] = data["total_views"] * 1_000_000_000

    if country_col:
        data["country"] = df[country_col].apply(normalize_country_code)
    else:
        data["country"] = "Unknown"

    if channel_col:
        data["channel_name"] = df[channel_col].astype(str).str.strip()
    else:
        data["channel_name"] = "Unknown Channel"

    if rank_col:
        data["global_rank"] = df[rank_col]
    else:
        data["global_rank"] = np.nan

    if description_col:
        data["description"] = df[description_col].astype(str)
    else:
        data["description"] = ""

    data = data.replace([np.inf, -np.inf], np.nan)

    data = data.dropna(subset=[
        "subscribers",
        "total_views",
        "total_videos",
    ])

    data = data[
        (data["subscribers"] >= 0) &
        (data["total_views"] >= 0) &
        (data["total_videos"] >= 0)
    ].copy()

    if data.empty:
        st.error("Dataset kosong setelah cleaning. Cek format angka pada kolom Subscribers, Total Views, dan Total Videos.")
        st.stop()

    tier_col = find_col(df, ["Tier"])

    if tier_col:
        data["tier"] = df.loc[data.index, tier_col].astype(str).str.strip()
    else:
        data["tier"] = data["subscribers"].apply(get_tier)

    data["tier_auto"] = data["subscribers"].apply(get_tier)

    data["views_per_subscriber"] = np.where(
        data["subscribers"] > 0,
        data["total_views"] / data["subscribers"],
        0
    )

    data["views_per_video"] = np.where(
        data["total_videos"] > 0,
        data["total_views"] / data["total_videos"],
        0
    )

    data["subscribers_million"] = data["subscribers"] / 1_000_000
    data["views_billion"] = data["total_views"] / 1_000_000_000

    data["log_subscribers"] = np.log1p(data["subscribers"])
    data["log_total_views"] = np.log1p(data["total_views"])

    data["country_label"] = data["country"].apply(country_label)

    return data


# ============================================================
# MODEL TRAINING
# ============================================================

@st.cache_resource
def train_model(data):
    features = ["subscribers", "total_videos", "country"]
    target = "total_views"

    X = data[features]
    y = np.log1p(data[target])

    numeric_features = ["subscribers", "total_videos"]
    categorical_features = ["country"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    if len(data) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    pipeline.fit(X_train, y_train)

    y_pred_log = pipeline.predict(X_test)
    y_test_real = np.expm1(y_test)
    y_pred_real = np.expm1(y_pred_log)

    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    r2 = r2_score(y_test_real, y_pred_real) if len(y_test_real) > 1 else 0

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    return pipeline, metrics


def get_feature_importance(pipeline):
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]

        numeric_features = ["subscribers", "total_videos"]

        encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        country_features = encoder.get_feature_names_out(["country"])

        feature_names = list(numeric_features) + list(country_features)
        importances = model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        })

        importance_df["Feature"] = importance_df["Feature"].apply(feature_label)

        importance_df = importance_df.sort_values(
            "Importance",
            ascending=False
        ).head(12)

        return importance_df

    except Exception:
        return pd.DataFrame(columns=["Feature", "Importance"])


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 📁 Dataset")
    st.caption("Dashboard akan otomatis membaca file `youtube_4k_channels_cleaned.csv` jika ada di Colab.")

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    detected_file = find_dataset_file()

    if uploaded is not None:
        df_raw = load_csv(uploaded)
        st.success("Dataset berhasil dibaca dari upload sidebar.")
    elif detected_file is not None:
        df_raw = load_csv(detected_file)
        st.success(f"Dataset otomatis dibaca dari: `{detected_file}`")
    else:
        st.warning("Dataset belum ditemukan. Upload CSV lewat sidebar atau letakkan file CSV di folder /content.")
        st.stop()

    st.markdown("---")
    st.markdown("### 🤖 Informasi Model")
    st.caption("Target prediksi: Total Views")
    st.caption("Input model: Subscribers, Total Videos, Country")
    st.caption("Algoritma: Random Forest Regression")


# ============================================================
# DATA + MODEL
# ============================================================

data = prepare_data(df_raw)
model, metrics = train_model(data)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">YouTube Channel Views Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Dashboard modern untuk menganalisis channel YouTube dan memprediksi Total Views berdasarkan Subscribers, Total Videos, dan Country.</div>',
    unsafe_allow_html=True
)


# ============================================================
# METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    metric_card(
        "Total Channel",
        format_number(len(data)),
        "Jumlah channel dalam dataset"
    )

with col2:
    metric_card(
        "Total Subscribers",
        format_number(data["subscribers"].sum()),
        "Akumulasi subscribers"
    )

with col3:
    metric_card(
        "Total Views",
        format_number(data["total_views"].sum()),
        "Akumulasi views"
    )

with col4:
    metric_card(
        "Rata-rata Views",
        format_number(data["total_views"].mean()),
        "Rata-rata views per channel"
    )

st.markdown("<hr>", unsafe_allow_html=True)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Overview",
    "📈 Visualisasi Data",
    "🤖 Prediksi Views",
    "🧾 Data Explorer"
])


# ============================================================
# TAB 1 - OVERVIEW
# ============================================================

with tab1:
    st.markdown('<div class="section-title">Overview Dataset</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Ringkasan data channel YouTube dan performa awal model Random Forest Regression.</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns([1.35, 1])

    with left:
        top_channels = data.sort_values("total_views", ascending=False).head(10)

        fig_top = px.bar(
            top_channels,
            x="total_views",
            y="channel_name",
            orientation="h",
            title="Top 10 Channel Berdasarkan Total Views",
            labels={
                "total_views": "Total Views",
                "channel_name": "Channel"
            },
            template="plotly_dark"
        )

        fig_top.update_layout(
            height=480,
            yaxis=dict(autorange="reversed"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title_font_size=20
        )

        st.plotly_chart(fig_top, use_container_width=True)

    with right:
        st.markdown("### Performa Model")

        st.metric("MAE", format_number(metrics["MAE"]))
        st.metric("RMSE", format_number(metrics["RMSE"]))
        st.metric("R² Score", f"{metrics['R2']:.4f}")

        st.info(
            "Model menggunakan Subscribers, Total Videos, dan Country. "
            "Target Total Views dilatih dalam bentuk log agar lebih stabil terhadap outlier."
        )

        tier_count = data["tier"].value_counts().reset_index()
        tier_count.columns = ["Tier", "Jumlah"]

        fig_tier = px.pie(
            tier_count,
            names="Tier",
            values="Jumlah",
            hole=0.55,
            title="Komposisi Tier Channel",
            template="plotly_dark"
        )

        fig_tier.update_layout(
            height=330,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig_tier, use_container_width=True)


# ============================================================
# TAB 2 - VISUALISASI DATA
# ============================================================

with tab2:
    st.markdown('<div class="section-title">Visualisasi Data</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Distribusi dibuat dengan log scale agar data besar dan outlier tetap mudah dibaca.</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        fig_sub = px.histogram(
            data,
            x="log_subscribers",
            nbins=35,
            title="Distribusi Subscribers (Log Scale)",
            labels={
                "log_subscribers": "Log Subscribers",
                "count": "Frekuensi"
            },
            template="plotly_dark"
        )

        fig_sub.update_layout(
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig_sub, use_container_width=True)

    with c2:
        fig_views = px.histogram(
            data,
            x="log_total_views",
            nbins=35,
            title="Distribusi Total Views (Log Scale)",
            labels={
                "log_total_views": "Log Total Views",
                "count": "Frekuensi"
            },
            template="plotly_dark"
        )

        fig_views.update_layout(
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig_views, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        scatter_data = data[
            (data["subscribers"] > 0) &
            (data["total_views"] > 0)
        ].copy()

        fig_scatter = px.scatter(
            scatter_data,
            x="subscribers",
            y="total_views",
            color="tier",
            hover_data=["channel_name", "country_label"],
            log_x=True,
            log_y=True,
            title="Hubungan Subscribers dan Total Views",
            labels={
                "subscribers": "Subscribers",
                "total_views": "Total Views",
                "tier": "Tier",
                "country_label": "Country"
            },
            template="plotly_dark"
        )

        fig_scatter.update_layout(
            height=460,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

    with c4:
        country_count = data["country_label"].value_counts().head(12).reset_index()
        country_count.columns = ["Country", "Jumlah Channel"]

        fig_country = px.bar(
            country_count,
            x="Jumlah Channel",
            y="Country",
            orientation="h",
            title="Top 12 Negara Berdasarkan Jumlah Channel",
            template="plotly_dark"
        )

        fig_country.update_layout(
            height=460,
            yaxis=dict(autorange="reversed"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig_country, use_container_width=True)

    st.markdown("### Feature Importance")

    importance_df = get_feature_importance(model)

    if not importance_df.empty:
        fig_importance = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Faktor yang Paling Berpengaruh pada Model",
            template="plotly_dark"
        )

        fig_importance.update_layout(
            height=470,
            yaxis=dict(autorange="reversed"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig_importance, use_container_width=True)
    else:
        st.warning("Feature importance belum bisa ditampilkan.")


# ============================================================
# TAB 3 - PREDIKSI
# ============================================================

with tab3:
    st.markdown('<div class="section-title">Prediksi Total Views Channel Baru</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Masukkan data channel baru untuk memprediksi Total Views. Tier dihitung otomatis dari jumlah subscribers.</div>',
        unsafe_allow_html=True
    )

    form_col, result_col = st.columns([1, 1.1])

    with form_col:
        st.markdown("### Input Data Channel")

        subscribers_input = st.number_input(
            "Jumlah Subscribers",
            min_value=0,
            value=100_000,
            step=10_000
        )

        total_videos_input = st.number_input(
            "Total Video",
            min_value=1,
            value=10,
            step=1
        )

        country_options = sorted(
            data["country"].dropna().unique().tolist(),
            key=lambda x: country_label(x)
        )

        default_country_index = 0
        for i, val in enumerate(country_options):
            if str(val).upper() == "ID" or str(val).lower() == "indonesia":
                default_country_index = i
                break

        country_input = st.selectbox(
            "Negara",
            options=country_options,
            index=default_country_index,
            format_func=country_label
        )

        tier_name, tier_class, tier_desc = get_tier_info(subscribers_input)

        st.markdown(f"""
        <div class="tier-box {tier_class}">
            Tier Otomatis: {tier_name}
        </div>
        <div style="font-size:13px; color:#cbd5e1; margin-top:8px;">
            {tier_desc}
        </div>
        """, unsafe_allow_html=True)

        predict_button = st.button(
            "Prediksi Total Views",
            use_container_width=True
        )

    with result_col:
        st.markdown("### Hasil Prediksi")

        if predict_button:
            input_df = pd.DataFrame({
                "subscribers": [subscribers_input],
                "total_videos": [total_videos_input],
                "country": [country_input],
            })

            pred_log = model.predict(input_df)[0]
            pred_views = np.expm1(pred_log)

            views_per_video = pred_views / total_videos_input if total_videos_input > 0 else 0
            views_per_subscriber = pred_views / subscribers_input if subscribers_input > 0 else 0

            st.markdown(f"""
            <div class="prediction-box">
                <div class="prediction-label">Estimasi Total Views</div>
                <div class="prediction-value">{format_number(pred_views)}</div>
                <div style="font-size:14px; margin-top:8px; opacity:0.95;">
                    ≈ {format_full_number(pred_views)} views
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")

            r1, r2 = st.columns(2)

            with r1:
                metric_card(
                    "Views per Video",
                    format_number(views_per_video),
                    "Estimasi rata-rata views per video"
                )

            with r2:
                metric_card(
                    "Views per Subscriber",
                    f"{views_per_subscriber:.2f}",
                    "Rasio views terhadap subscriber"
                )
            st.markdown("")
            st.success("Prediksi berhasil dibuat. Nilai ini merupakan estimasi berdasarkan pola dataset.")

        else:
            st.markdown("""
            <div class="card">
                Isi data channel di sebelah kiri, lalu klik tombol <b>Prediksi Total Views</b>.
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Contoh Input Demo")

    demo_df = pd.DataFrame({
        "Skenario": [
            "Channel kecil Indonesia",
            "Channel menengah Indonesia",
            "Channel besar global"
        ],
        "Subscribers": [
            "100.000",
            "1.000.000",
            "10.000.000"
        ],
        "Total Video": [
            "10",
            "250",
            "1.000"
        ],
        "Negara": [
            "ID - Indonesia",
            "ID - Indonesia",
            "US - United States"
        ],
        "Keterangan": [
            "Demo channel baru / under 1M",
            "Demo channel berkembang",
            "Demo channel besar"
        ]
    })

    st.dataframe(demo_df, use_container_width=True, hide_index=True)


# ============================================================
# TAB 4 - DATA EXPLORER
# ============================================================

with tab4:
    st.markdown('<div class="section-title">Data Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Lihat data yang sudah dibersihkan dan digunakan oleh model.</div>',
        unsafe_allow_html=True
    )

    f1, f2 = st.columns(2)

    with f1:
        selected_tier = st.multiselect(
            "Filter Tier",
            options=sorted(data["tier"].unique().tolist()),
            default=sorted(data["tier"].unique().tolist())
        )

    with f2:
        country_filter_options = sorted(
            data["country"].dropna().unique().tolist(),
            key=lambda x: country_label(x)
        )

        selected_country = st.multiselect(
            "Filter Negara",
            options=country_filter_options,
            default=[],
            format_func=country_label
        )

    filtered = data.copy()

    if selected_tier:
        filtered = filtered[filtered["tier"].isin(selected_tier)]

    if selected_country:
        filtered = filtered[filtered["country"].isin(selected_country)]

    display_df = filtered[[
        "global_rank",
        "channel_name",
        "country_label",
        "tier",
        "subscribers",
        "total_videos",
        "total_views",
        "views_per_subscriber",
        "views_per_video"
    ]].copy()

    display_df = display_df.rename(columns={
        "global_rank": "Global Rank",
        "channel_name": "Channel Name",
        "country_label": "Country",
        "tier": "Tier",
        "subscribers": "Subscribers",
        "total_videos": "Total Videos",
        "total_views": "Total Views",
        "views_per_subscriber": "Views per Subscriber",
        "views_per_video": "Views per Video"
    })

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    csv_data = display_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Data Filtered CSV",
        data=csv_data,
        file_name="youtube_dashboard_filtered.csv",
        mime="text/csv",
        use_container_width=True
    )
