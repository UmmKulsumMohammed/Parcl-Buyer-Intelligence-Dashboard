import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Parcl Buyer Intelligence Dashboard",
    layout="wide"
)

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv("buyer_segments_output.csv")

df = load_data()

# ==========================================
# TITLE
# ==========================================

st.title("🏠 Parcl Buyer Intelligence Dashboard")
st.markdown(
"""
Machine Learning Based Buyer Segmentation and Investment Profiling
for Real Estate Market Intelligence
"""
)

# ==========================================
# SIDEBAR FILTERS
# ==========================================

st.sidebar.header("Filters")

country = st.sidebar.selectbox(
    "Country",
    ["All"] + sorted(df["country"].dropna().unique().tolist())
)

region = st.sidebar.selectbox(
    "Region",
    ["All"] + sorted(df["region"].dropna().unique().tolist())
)

purpose = st.sidebar.selectbox(
    "Acquisition Purpose",
    ["All"] + sorted(df["acquisition_purpose"].dropna().unique().tolist())
)

client_type = st.sidebar.selectbox(
    "Client Type",
    ["All"] + sorted(df["client_type"].dropna().unique().tolist())
)

# ==========================================
# FILTER DATA
# ==========================================

filtered_df = df.copy()

if country != "All":
    filtered_df = filtered_df[
        filtered_df["country"] == country
    ]

if region != "All":
    filtered_df = filtered_df[
        filtered_df["region"] == region
    ]

if purpose != "All":
    filtered_df = filtered_df[
        filtered_df["acquisition_purpose"] == purpose
    ]

if client_type != "All":
    filtered_df = filtered_df[
        filtered_df["client_type"] == client_type
    ]

# ==========================================
# EMPTY DATA CHECK
# ==========================================

if filtered_df.empty:
    st.warning(
        "No records found for selected filters. Please choose different filters."
    )
    st.stop()

# ==========================================
# OVERVIEW METRICS
# ==========================================

st.header("Buyer Segmentation Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Buyers",
    len(filtered_df)
)

col2.metric(
    "Countries",
    filtered_df["country"].nunique()
)

col3.metric(
    "Regions",
    filtered_df["region"].nunique()
)

# ==========================================
# CLUSTER DISTRIBUTION
# ==========================================

cluster_counts = (
    filtered_df["Cluster"]
    .value_counts()
    .sort_index()
)

cluster_df = pd.DataFrame({
    "Cluster": cluster_counts.index.astype(str),
    "Buyer Count": cluster_counts.values
})

fig = px.bar(
    cluster_df,
    x="Cluster",
    y="Buyer Count",
    title="Cluster Distribution"
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================
# INVESTOR BEHAVIOR DASHBOARD
# ==========================================

st.header("Investor Behavior Dashboard")

col1, col2 = st.columns(2)

purpose_counts = (
    filtered_df["acquisition_purpose"]
    .value_counts()
    .reset_index()
)

purpose_counts.columns = [
    "Purpose",
    "Count"
]

fig1 = px.pie(
    purpose_counts,
    names="Purpose",
    values="Count",
    title="Acquisition Purpose"
)

col1.plotly_chart(
    fig1,
    use_container_width=True
)

loan_counts = (
    filtered_df["loan_applied"]
    .value_counts()
    .reset_index()
)

loan_counts.columns = [
    "Loan Applied",
    "Count"
]

fig2 = px.pie(
    loan_counts,
    names="Loan Applied",
    values="Count",
    title="Loan Behaviour"
)

col2.plotly_chart(
    fig2,
    use_container_width=True
)

# ==========================================
# GEOGRAPHIC ANALYSIS
# ==========================================

st.header("Geographic Buyer Analysis")

country_counts = (
    filtered_df["country"]
    .value_counts()
    .reset_index()
)

country_counts.columns = [
    "Country",
    "Count"
]

fig3 = px.bar(
    country_counts,
    x="Country",
    y="Count",
    title="Country Distribution"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

region_counts = (
    filtered_df["region"]
    .value_counts()
    .reset_index()
)

region_counts.columns = [
    "Region",
    "Count"
]

fig4 = px.bar(
    region_counts,
    x="Region",
    y="Count",
    title="Region Distribution"
)

st.plotly_chart(
    fig4,
    use_container_width=True
)

# ==========================================
# SEGMENT INSIGHTS PANEL
# ==========================================

st.header("Segment Insights Panel")

summary = (
    filtered_df
    .groupby("Cluster")
    [
        [
            "age",
            "sale_price",
            "floor_area_sqft",
            "satisfaction_score"
        ]
    ]
    .mean()
    .round(2)
)

st.dataframe(
    summary,
    use_container_width=True
)

# ==========================================
# RAW DATA
# ==========================================

st.header("Clustered Buyer Dataset")

st.dataframe(
    filtered_df,
    use_container_width=True
)
