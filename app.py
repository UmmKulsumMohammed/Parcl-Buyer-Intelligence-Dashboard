import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Buyer Segmentation Dashboard",
    layout="wide"
)

# =====================================
# LOAD DATA
# =====================================

@st.cache_data
def load_data():
    return pd.read_csv("buyer_segments_output.csv")

df = load_data()

# =====================================
# TITLE
# =====================================

st.title("🏠 Buyer Segmentation & Investment Profiling Dashboard")

st.markdown(
"""
Machine Learning Based Buyer Segmentation and Investment Profiling
for Real Estate Market Intelligence
"""
)

# =====================================
# SIDEBAR FILTERS
# =====================================

st.sidebar.header("Filters")

country = st.sidebar.selectbox(
    "Country",
    ["All"] + sorted(df["country"].unique().tolist())
)

region = st.sidebar.selectbox(
    "Region",
    ["All"] + sorted(df["region"].unique().tolist())
)

purpose = st.sidebar.selectbox(
    "Acquisition Purpose",
    ["All"] + sorted(df["acquisition_purpose"].unique().tolist())
)

client_type = st.sidebar.selectbox(
    "Client Type",
    ["All"] + sorted(df["client_type"].unique().tolist())
)

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

# =====================================
# OVERVIEW
# =====================================

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

cluster_counts = (
    filtered_df["Cluster"]
    .value_counts()
    .sort_index()
)

fig = px.bar(
    x=cluster_counts.index,
    y=cluster_counts.values,
    labels={
        "x":"Cluster",
        "y":"Buyer Count"
    },
    title="Cluster Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================
# INVESTOR BEHAVIOR
# =====================================

st.header("Investor Behavior Dashboard")

col1, col2 = st.columns(2)

purpose_counts = (
    filtered_df["acquisition_purpose"]
    .value_counts()
)

fig1 = px.pie(
    values=purpose_counts.values,
    names=purpose_counts.index,
    title="Acquisition Purpose"
)

col1.plotly_chart(
    fig1,
    use_container_width=True
)

loan_counts = (
    filtered_df["loan_applied"]
    .value_counts()
)

fig2 = px.pie(
    values=loan_counts.values,
    names=loan_counts.index,
    title="Loan Behavior"
)

col2.plotly_chart(
    fig2,
    use_container_width=True
)

# =====================================
# GEOGRAPHIC ANALYSIS
# =====================================

st.header("Geographic Buyer Analysis")

country_counts = (
    filtered_df["country"]
    .value_counts()
)

fig = px.bar(
    x=country_counts.index,
    y=country_counts.values,
    title="Country Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

region_counts = (
    filtered_df["region"]
    .value_counts()
)

fig = px.bar(
    x=region_counts.index,
    y=region_counts.values,
    title="Region Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================
# SEGMENT INSIGHTS
# =====================================

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

# =====================================
# RAW DATA
# =====================================

st.header("Clustered Dataset")

st.dataframe(
    filtered_df,
    use_container_width=True
)