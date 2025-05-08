import streamlit as st
import joblib
import pandas as pd
import pymongo
from openai import OpenAI
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- Load Data and Model
data_file_path = r"focused_synthetic_loan_data.csv"
loan_data = pd.read_csv(data_file_path)

predictors2 = [
    'TotalDebtToIncomeRatio', 'MonthlyIncome', 'LoanAmount', 'InterestRate',
    'MonthlyLoanPayment', 'LengthOfCreditHistory', 'TotalAssets', 'CreditScore', 'Age'
]
ml_model = joblib.load("loan_model_3.pkl")

# --- MongoDB
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["Project"]
mongo_collection = mongo_db["loan_approval"]

# --- OpenAI Setup
client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=""
)

# --- Loan Prediction Logic
def predict_loan(input_data):
    df = pd.DataFrame([input_data], columns=predictors2)
    prediction = ml_model.predict(df)
    return "✅ Approved" if prediction[0] == 1 else "❌ Not Approved"

# --- Custom Queries
def handle_custom_query(nlq):
    if "top 10 loan defaulters" in nlq.lower():
        if 'LoanAmount' in loan_data.columns:
            top_defaulters = loan_data[loan_data['LoanApproved'] == 0].sort_values(by='LoanAmount', ascending=False).head(10)
            return top_defaulters[['CustomerID', 'LoanAmount']] if 'CustomerID' in loan_data.columns else top_defaulters
        else:
            return "❗ 'LoanAmount' column not found."

    elif "average credit score" in nlq.lower():
        if 'CreditScore' in loan_data.columns:
            avg_score = loan_data['CreditScore'].mean()
            return f"📊 Average Credit Score: {avg_score:.2f}"
        else:
            return "❗ 'CreditScore' column not found."

    elif "compare loan default rates across different income groups" in nlq.lower() or "default rates across income groups" in nlq.lower():
        pipeline = [
            {
                "$addFields": {
                    "IncomeGroup": {
                        "$switch": {
                            "branches": [
                                { "case": { "$lte": ["$MonthlyIncome", 20000] }, "then": "Low" },
                                { "case": { "$and": [ { "$gt": ["$MonthlyIncome", 20000] }, { "$lte": ["$MonthlyIncome", 50000] } ] }, "then": "Lower-Middle" },
                                { "case": { "$and": [ { "$gt": ["$MonthlyIncome", 50000] }, { "$lte": ["$MonthlyIncome", 80000] } ] }, "then": "Middle" },
                                { "case": { "$and": [ { "$gt": ["$MonthlyIncome", 80000] }, { "$lte": ["$MonthlyIncome", 120000] } ] }, "then": "Upper-Middle" }
                            ],
                            "default": "High"
                        }
                    }
                }
            },
            {
                "$group": {
                    "_id": "$IncomeGroup",
                    "defaultRate": {
                        "$avg": { "$cond": [ { "$eq": ["$LoanApproved", 0] }, 1, 0 ] }
                    },
                    "total": { "$sum": 1 }
                }
            },
            {
                "$sort": { "_id": 1 }
            }
        ]

        results = list(mongo_collection.aggregate(pipeline))
        df = pd.DataFrame(results)
        df["defaultRate"] = (df["defaultRate"] * 100).round(2)

        st.write("### 💡 Default Rates by Income Group")
        st.dataframe(df.rename(columns={
            "_id": "Income Group",
            "defaultRate": "Default Rate (%)",
            "total": "Total Loans"
        }))

        st.write("### 📊 Default Rate Bar Chart")
        st.bar_chart(df.set_index('_id')['defaultRate'])

        return ""

    return None

# --- Visual Chart Handler
def handle_visualization_request(nlq):
    try:
        data = pd.DataFrame(list(mongo_collection.find()))
        if data.empty:
            return "❗ No data available from MongoDB."
        if "_id" in data.columns:
            data.drop(columns=["_id"], inplace=True)

        requested_columns = []
        for col in data.columns:
            if col.lower() in nlq.lower():
                requested_columns.append(col)

        if not requested_columns:
            return "❗ No matching columns found in the query."

        chart_type = None
        if "bar chart" in nlq.lower():
            chart_type = "bar"
        elif "pie chart" in nlq.lower():
            chart_type = "pie"
        elif "line chart" in nlq.lower():
            chart_type = "line"
        else:
            return "❗ Please specify the chart type (bar, pie, or line)."

        for col in requested_columns:
            fig, ax = plt.subplots()
            if chart_type == "bar":
                data[col].value_counts().sort_index().plot(kind="bar", ax=ax)
            elif chart_type == "pie":
                data[col].value_counts().plot(kind="pie", autopct='%1.1f%%', ax=ax)
            elif chart_type == "line":
                data[col].plot(kind="line", ax=ax)

            ax.set_title(f"{chart_type.title()} Chart for {col}")
            st.pyplot(fig)

        return "✅ Visualizations rendered successfully."
    except Exception as e:
        return f"Error: {e}"

# --- Outlier Detection
def detect_outliers(data):
    numeric_columns = data.select_dtypes(include=np.number).columns
    if numeric_columns.empty:
        st.warning("No numeric columns for outlier detection.")
        return

    for column in numeric_columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = data[(data[column] < lower) | (data[column] > upper)]

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(x=data[column], ax=ax, color="skyblue", flierprops={"marker": "o", "color": "red"})
        ax.set_title(f"Outliers in {column}")
        st.pyplot(fig)

        if not outliers.empty:
            st.write(f"Outliers in {column}:")
            st.dataframe(outliers)

# --- Main Natural Language Handler
def process_query(nlq):
    try:
        if "detect outliers" in nlq.lower():
            detect_outliers(loan_data)
            return "✅ Outlier detection complete. See visualizations above."

        if any(kw in nlq.lower() for kw in ["bar chart", "pie chart", "line chart"]):
            return handle_visualization_request(nlq)

        custom = handle_custom_query(nlq)
        if custom is not None:
            return custom

        prompt = f"""You are a data analyst with access to the full loan dataset.
Answer this question based on the full dataset:
{nlq}"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for data analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1000
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# --- Streamlit Layout
st.set_page_config(page_title="Loan App: Predictor & Q&A", layout="centered")
st.title("🏦 Loan Approval Predictor & AI Data Explorer")
st.markdown("Enter details to predict loan approval or ask a question below 👇")

# --- Input Form
col1, col2 = st.columns(2)
with col1:
    tdti = st.number_input("Total Debt to Income Ratio", 0.0, 1.0, step=0.01)
    income = st.number_input("Monthly Income (₹)", 0.0, step=100.0)
    loan_amt = st.number_input("Loan Amount (₹)", 0.0, step=1000.0)
    interest = st.number_input("Interest Rate", 0.0, step=0.01)
    emi = st.number_input("Monthly EMI (₹)", 0.0, step=100.0)

with col2:
    credit_hist = st.number_input("Credit History (Years)", 0, 50)
    assets = st.number_input("Total Assets (₹)", 0.0, step=1000.0)
    credit_score = st.number_input("Credit Score", 0, 900)
    age = st.number_input("Age (Years)", 0, 100)

if st.button("🔍 Predict Loan"):
    input_data = {
        'TotalDebtToIncomeRatio': tdti,
        'MonthlyIncome': income,
        'LoanAmount': loan_amt,
        'InterestRate': interest,
        'MonthlyLoanPayment': emi,
        'LengthOfCreditHistory': credit_hist,
        'TotalAssets': assets,
        'CreditScore': credit_score,
        'Age': age
    }
    if tdti > 0.6 and credit_score < 670:
        result = "❌ Not Approved (manual rule: risky profile)"
    else:
        result = predict_loan(input_data)
    st.success(f"📊 Prediction: {result}")

# --- Chat UI
st.header("💬 Ask Questions About the Dataset")
nlq = st.text_input("Ask a question:")
if st.button("Send"):
    if nlq.strip():
        response = process_query(nlq)
        st.write("### 💡 Answer:")
        st.write(response)
    else:
        st.warning("Please enter a question.")
