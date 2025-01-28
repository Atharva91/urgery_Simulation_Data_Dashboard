# AI-Powered Surgery Simulation Data Dashboard

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import openai
from pyod.models.iforest import IForest
from prophet import Prophet

# Generate Simulation Data
def generate_dummy_data():
    np.random.seed(42)
    timestamps = [datetime(2023, 1, 1, np.random.randint(0, 24), np.random.randint(0, 60)) + timedelta(days=i) for i in range(500)]
    data = {
        "SessionID": [f"S{i+1}" for i in range(500)],
        "SurgeonID": np.random.randint(1, 51, size=500),
        "TaskID": np.random.randint(1, 11, size=500),
        "CompletionTime": np.random.uniform(5, 30, size=500),
        "ErrorRate": np.random.uniform(0, 0.2, size=500),
        "ToolUsage": np.random.uniform(1, 5, size=500),
        "EnergyConsumption": np.random.uniform(50, 150, size=500),
        "Accuracy": np.random.uniform(0.8, 1.0, size=500),
        "Timestamp": timestamps
    }
    return pd.DataFrame(data)

data = generate_dummy_data()

# Streamlit App Setup
st.set_page_config(page_title="Surgery Simulation Dashboard", layout="wide")
st.sidebar.header("Navigation")
selected_page = st.sidebar.radio("Go to", ["Dashboard Overview", "Detailed Analytics", "Predictive Analytics", "Feedback"])

# Filters
st.sidebar.header("Filters")
selected_surgeon = st.sidebar.multiselect("Select Surgeon(s)", options=list(data["SurgeonID"].unique()), default=data["SurgeonID"].unique())
selected_task = st.sidebar.multiselect("Select Task(s)", options=list(data["TaskID"].unique()), default=data["TaskID"].unique())
time_range = st.sidebar.slider("Select Completion Time Range (minutes)", min_value=float(data["CompletionTime"].min()), max_value=float(data["CompletionTime"].max()), value=(float(data["CompletionTime"].min()), float(data["CompletionTime"].max())))

# Apply Filters
filtered_data = data[data["SurgeonID"].isin(selected_surgeon)]
filtered_data = filtered_data[filtered_data["TaskID"].isin(selected_task)]
filtered_data = filtered_data[(filtered_data["CompletionTime"] >= time_range[0]) & (filtered_data["CompletionTime"] <= time_range[1])]

# AI Components
# 1. Anomaly Detection
def detect_anomalies(data):
    features = data[["CompletionTime", "ErrorRate", "ToolUsage", "EnergyConsumption", "Accuracy"]]
    model = IForest(random_state=42)
    model.fit(features)
    data["Anomaly"] = model.predict(features)  # 1 = anomaly, 0 = normal
    return data, model

# 2. Performance Scoring
def score_performance(data):
    data["PerformanceScore"] = (
        1.0 / data["CompletionTime"] * 0.4 +
        (1.0 - data["ErrorRate"]) * 0.4 +
        data["Accuracy"] * 0.2
    ) * 100  # Normalize to a percentage scale
    return data

# 3. Natural Language Insights
def generate_insight(prompt):
    openai.api_key = "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"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100
    )
    return response.choices[0].text.strip()

# 4. Forecasting Energy Consumption
def forecast_energy(data):
    df = data[["Timestamp", "EnergyConsumption"]].rename(columns={"Timestamp": "ds", "EnergyConsumption": "y"})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast

# Dashboard Overview
if selected_page == "Dashboard Overview":
    st.title("AI-Powered Surgery Simulation Data Dashboard")
    st.write("## Business Context and Motivation")
    st.markdown(
        """
        Robotic-assisted surgeries are transforming the medical landscape by enhancing precision and reducing recovery times. 
        This dashboard is designed to analyze and improve surgeon performance by providing actionable insights, identifying areas for improvement, and predicting outcomes.
        """
    )

    st.write("## Filtered Simulation Data")
    filtered_data = score_performance(filtered_data)
    st.dataframe(filtered_data, use_container_width=True)

    st.write("## Overview of Metrics")
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Completion Time Distribution")
        fig = px.histogram(filtered_data, x="CompletionTime", nbins=20, title="Completion Time Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.write("### Performance Scores")
        fig = px.histogram(filtered_data, x="PerformanceScore", nbins=20, title="Performance Score Distribution", color="SurgeonID")
        st.plotly_chart(fig, use_container_width=True)

# Detailed Analytics
elif selected_page == "Detailed Analytics":
    st.title("Detailed Analytics")

    st.write("### Tool Usage Trends")
    fig = px.line(filtered_data, x="Timestamp", y="ToolUsage", title="Tool Usage Over Time", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.write("### Anomaly Detection")
    anomaly_data, anomaly_model = detect_anomalies(filtered_data)
    fig = px.scatter(anomaly_data, x="CompletionTime", y="ErrorRate", color="Anomaly", title="Anomaly Detection", hover_data=["SurgeonID", "TaskID"])
    st.plotly_chart(fig, use_container_width=True)
    st.write("#### Anomaly Summary")
    st.write(anomaly_data[anomaly_data["Anomaly"] == 1].describe())

# Predictive Analytics
elif selected_page == "Predictive Analytics":
    st.title("Predictive Analytics")
    st.markdown(
        """
        Predict and optimize surgical simulation outcomes using AI. This section focuses on identifying task IDs based on performance metrics and estimating completion times.
        """
    )

    def train_predictive_model(data):
        X = data[["CompletionTime", "ErrorRate", "ToolUsage", "EnergyConsumption"]]
        y = data["TaskID"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return model, mse

    model, mse = train_predictive_model(filtered_data)
    st.write(f"Model trained with Mean Squared Error: {mse:.2f}")

    user_completion_time = st.number_input("Enter Completion Time (minutes):", min_value=0.0, step=0.1)
    user_error_rate = st.number_input("Enter Error Rate (0 to 1):", min_value=0.0, max_value=1.0, step=0.01)
    user_tool_usage = st.number_input("Enter Tool Usage (1 to 5):", min_value=1.0, max_value=5.0, step=0.1)
    user_energy_consumption = st.number_input("Enter Energy Consumption (50 to 150):", min_value=50.0, max_value=150.0, step=1.0)

    if st.button("Predict Task ID"):
        user_input = np.array([[user_completion_time, user_error_rate, user_tool_usage, user_energy_consumption]])
        prediction = model.predict(user_input)
        st.write(f"Predicted Task ID: {int(round(prediction[0]))}")

    st.write("### Energy Consumption Forecast")
    forecast = forecast_energy(filtered_data)
    fig = px.line(forecast, x="ds", y="yhat", title="Energy Consumption Forecast")
    st.plotly_chart(fig, use_container_width=True)

# Feedback
elif selected_page == "Feedback":
    st.title("Feedback")
    st.write("Your feedback helps us improve the dashboard!")

    feedback_text = st.text_area("Provide your feedback:")
    if st.button("Submit Feedback"):
        st.write("Thank you for your valuable input!")

    st.write("### Download Assessment")
    st.download_button("Download Filtered Data", data.to_csv(index=False), "filtered_data.csv", "text/csv")

    st.write("### Potential Enhancements")
    st.markdown("- **Real-time integration:** Enable live updates from surgical tools.")
    st.markdown("- **Enhanced visualizations:** Add dynamic interactivity and more KPIs.")
    st.markdown("- **Model accuracy improvements:** Incorporate additional performance metrics.")
