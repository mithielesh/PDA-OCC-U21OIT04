import streamlit as st
from PIL import Image
import pandas as pd

customer_df = pd.read_csv("data/processed/customer_features.csv")

st.set_page_config(layout="wide")

st.title("Analytics Intelligence Framework")

st.markdown("---")

# =========================
# KPI ROW
# =========================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Customers", f"{len(customer_df)}")
col2.metric("Total Revenue", f"${customer_df['Total_Spent'].sum():,.0f}")
col3.metric("Avg Spend", f"${customer_df['Total_Spent'].mean():,.0f}")
col4.metric("High Value %", f"{customer_df['High_Value_Customer'].mean()*100:.1f}%")

st.markdown("---")

# =========================
# MAIN VISUAL GRID
# =========================

left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.subheader("Revenue Distribution")
    st.image("images/revenue_distribution.png", use_container_width=True)

with right_col:
    st.subheader("Top 5 Customers")
    top5 = customer_df.sort_values("Total_Spent", ascending=False).head(5)
    st.dataframe(top5, use_container_width=True, height=260)

st.markdown("")

# =========================
# BOTTOM GRID
# =========================

colA, colB, colC = st.columns(3)

with colA:
    st.subheader("High Value Split")
    st.image("images/high_value_split.png", use_container_width=True)

with colB:
    st.subheader("Regression Feature Importance")
    st.image("images/regression_feature_importance.png", use_container_width=True)

with colC:
    st.subheader("Classification Feature Importance")
    st.image("images/classification_feature_importance.png", use_container_width=True)
