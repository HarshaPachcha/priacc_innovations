# dashboard_app.py
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ----------------------------------------------------------
# Streamlit Page Config
# ----------------------------------------------------------
st.set_page_config(page_title="Predictive Intelligence Dashboard", layout="wide")
st.title("📊 Predictive Intelligence Dashboard (IEAP)")
st.sidebar.header("🧠 Employee Attrition Prediction")

# ----------------------------------------------------------
# Load Datasets (from your given paths)
# ----------------------------------------------------------
@st.cache_data
def load_data():
    attrition = pd.read_csv("C:\\Users\\harsh\\OneDrive\\Desktop\\CSV\\Employee Attrition.csv")
    fraud = pd.read_csv("C:\\Users\\harsh\\OneDrive\\Desktop\\CSV\\Fraud Detection.csv")
    sales = pd.read_csv(
        r"C:\Users\harsh\OneDrive\Desktop\CSV\Superstore Sales.csv",
        parse_dates=['Order Date'],
        dayfirst=True
    )
    return attrition, fraud, sales

try:
    attrition_df, fraud_df, sales_df = load_data()
    st.success("✅ Datasets loaded successfully!")
except FileNotFoundError as e:
    st.error(f"❌ Dataset not found: {e}")
    st.stop()

# ----------------------------------------------------------
# Load Machine Learning Model
# ----------------------------------------------------------
try:
    model = joblib.load("attrition_model.pkl")
except FileNotFoundError:
    st.error("❌ Model file 'attrition_model.pkl' not found. Please train and save it first.")
    st.stop()

# ----------------------------------------------------------
# Sidebar — Employee Attrition Input Fields
# ----------------------------------------------------------
st.sidebar.subheader("Enter Employee Details")
age = st.sidebar.slider("Age", 18, 60, 30)
income = st.sidebar.number_input("Monthly Income", 1000, 50000, 5000)
overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])

# Prepare DataFrame for Prediction
input_data = pd.DataFrame(
    [[age, income, 1 if overtime == "Yes" else 0]],
    columns=["Age", "MonthlyIncome", "OverTime"]
)

# ----------------------------------------------------------
# Prediction Section
# ----------------------------------------------------------
if st.sidebar.button("Predict Attrition"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if pred == 1:
        st.error(f"🚨 Likely to Leave (Attrition Probability: {prob*100:.2f}%)")
    else:
        st.success(f"✅ Likely to Stay (Retention Probability: {(1-prob)*100:.2f}%)")

    # --- Plotly Bar Chart for Probability ---
    fig = px.bar(
        x=["Attrition Risk (%)", "Retention Chance (%)"],
        y=[prob * 100, (1 - prob) * 100],
        color=["Attrition Risk (%)", "Retention Chance (%)"],
        color_discrete_map={
            "Attrition Risk (%)": "red",
            "Retention Chance (%)": "green"
        },
        text=[f"{prob*100:.2f}%", f"{(1-prob)*100:.2f}%"],
        title="Employee Attrition Probability"
    )
    fig.update_traces(textposition="outside", textfont_size=14)
    fig.update_layout(yaxis_title="Percentage", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# Data Insights & Visualization
# ----------------------------------------------------------
st.markdown("---")
st.header("📈 Data Insights and Analysis")

tab1, tab2, tab3 = st.tabs(["👤 Employee Attrition", "💳 Fraud Detection", "💼 Sales Overview"])

# 1️⃣ Employee Attrition Visuals
with tab1:
    if "Age" in attrition_df.columns and "Attrition" in attrition_df.columns:
        fig_age = px.histogram(
            attrition_df, x="Age", color="Attrition",
            title="Attrition Distribution by Age", nbins=20
        )
        st.plotly_chart(fig_age, use_container_width=True)

    if "MonthlyIncome" in attrition_df.columns:
        fig_income = px.box(
            attrition_df, y="MonthlyIncome", color="Attrition",
            title="Income vs Attrition"
        )
        st.plotly_chart(fig_income, use_container_width=True)

# 2️⃣ Fraud Detection Visuals
with tab2:
    # possible column name variants
    amount_col_candidates = ["TransactionAmount", "amount", "Amount", "transaction_amount"]
    target_col_candidates = ["IsFraud", "isFraud", "is_fraud", "Is_Fraud"]

    # find available columns
    amount_col = next((c for c in amount_col_candidates if c in fraud_df.columns), None)
    target_col = next((c for c in target_col_candidates if c in fraud_df.columns), None)

    if amount_col is not None and target_col is not None:
        # make a copy to avoid modifying original df
        plot_df = fraud_df[[amount_col, target_col]].copy()

        # ensure target is categorical / string for color (avoids continuous-color issues)
        plot_df[target_col] = plot_df[target_col].astype(str)

        fig_fraud = px.histogram(
            plot_df,
            x=amount_col,
            color=target_col,
            nbins=50,
            title="Fraudulent vs Legitimate Transactions (by Amount)",
            labels={amount_col: "Transaction Amount", target_col: "Is Fraud?"}
        )
        fig_fraud.update_layout(barmode="overlay")   # overlay to compare distributions
        fig_fraud.update_traces(opacity=0.7)

        st.plotly_chart(fig_fraud, use_container_width=True)
    else:
        st.warning("Required columns for fraud histogram not found. Expecting amount and isFraud (or TransactionAmount/IsFraud).")


# 3️⃣ Sales Overview
with tab3:
    if "Order Date" in sales_df.columns and "Sales" in sales_df.columns:
        monthly_sales = sales_df.groupby(pd.Grouper(key="Order Date", freq="M"))["Sales"].sum().reset_index()
        fig_sales = px.line(
            monthly_sales, x="Order Date", y="Sales",
            markers=True, title="Monthly Sales Trend"
        )
        st.plotly_chart(fig_sales, use_container_width=True)

    if "Category" in sales_df.columns:
        cat_sales = sales_df.groupby("Category")["Sales"].sum().reset_index()
        fig_cat = px.pie(
            cat_sales, names="Category", values="Sales",
            title="Sales Share by Category"
        )
        st.plotly_chart(fig_cat, use_container_width=True)
