PUBLISHER:
import paho.mqtt.client as mqtt
import random
import time
import json

# MQTT Broker Details
broker = "test.mosquitto.org"  # Public MQTT Broker
topic = "smart_inventory/data"

# Initialize MQTT Client
client = mqtt.Client()
client.connect(broker, 1883, 60)

while True:
    reorder_level = random.randint(30, 80)  # Random reorder threshold
    current_stock = random.randint(reorder_level + 5, 150)  # Ensures stock is above reorder level

    data = {
        "product_key": random.randint(1, 5),  # Assuming 5 products in DB
        "current_stock": current_stock,
        "reorder_level": reorder_level,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    }

    # Publish Data
    client.publish(topic, json.dumps(data))
    print(f"📤 Published: {data}")

    time.sleep(3)  # Faster simulation (every 3 seconds)


SUBSCRIBER:
import paho.mqtt.client as mqtt
import mysql.connector
import json

# MySQL Connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="VIJAY@25",   
    database="SmartInventory"
)
cursor = db.cursor()

# MQTT Settings
broker = "test.mosquitto.org"
topic = "smart_inventory/data"

def on_message(client, userdata, message):
    data = json.loads(message.payload.decode())

    # Check for duplicate timestamps
    query_check = "SELECT COUNT(*) FROM InventoryData WHERE timestamp = %s"
    cursor.execute(query_check, (data["timestamp"],))
    (count,) = cursor.fetchone()

    if count == 0:  # Only insert if timestamp is unique
        query = """
        INSERT INTO InventoryData (product_key, current_stock, reorder_level, timestamp)
        VALUES (%s, %s, %s, %s)
        """
        values = (data["product_key"], data["current_stock"], data["reorder_level"], data["timestamp"])
        cursor.execute(query, values)
        db.commit()
        print(f"✅ Inserted into MySQL: {values}")
    else:
        print("⚠ Skipped duplicate entry.")

# Setup MQTT Subscriber
client = mqtt.Client()
client.connect(broker, 1883, 60)
client.subscribe(topic)
client.on_message = on_message

print(f"📡 Listening to MQTT Topic: {topic}")
client.loop_forever()


APP:
import streamlit as st
import mysql.connector
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import requests
import yagmail
from datetime import datetime

# === 🔹 Load ML Models ===
arima_model = joblib.load("C:/Users/vijayabalan/Downloads/arima_model.pkl")
iso_forest = joblib.load("C:/Users/vijayabalan/Downloads/isolation_forest.pkl")
q_table = joblib.load("C:/Users/vijayabalan/Downloads/optimized_q_table (1).pkl")
collab_recommendations = pd.read_csv("C:/Users/vijayabalan/Downloads/Collaborative_Filtering_Recommendations_Fixed.csv")
content_recommendations = pd.read_csv("C:/Users/vijayabalan/Downloads/Content_Based_Recommendations_Fixed.csv")

# === 🔹 MySQL Connection ===
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="VIJAY@25",
        database="SmartInventory"
    )

# === 🔹 Telegram Notification ===
TELEGRAM_API_TOKEN = "7932666381:AAEDrkX8mbAjjuHww2vfDxgdKq-UEisOK_8"
CHAT_ID = "7442967349"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_API_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    response = requests.post(url, data=data)
    return response.status_code == 200

# === 🔹 Email Notification ===
def send_email(subject, message, recipient_email):
    sender_email = "vijayabalan25032001v@gmail.com"
    sender_password = "hqpk qokl naob gxnb"
    yag = yagmail.SMTP(user=sender_email, password=sender_password)
    yag.send(to=recipient_email, subject=subject, contents=message)
    return True

# === 🔹 Fetch IoT Data ===
def fetch_latest_data():
    db = get_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM InventoryData ORDER BY timestamp DESC LIMIT 50")
    data = cursor.fetchall()
    db.close()
    return pd.DataFrame(data) if data else pd.DataFrame()

# === 🔹 Constants for Stock Classification ===
REORDER_LEVEL = 15
OVERSTOCK_LEVEL = 40
ACTIONS = ["No Restock", "Medium Restock", "Large Restock"]

def classify_state(stock):
    if stock <= REORDER_LEVEL:
        return 0  # Low Stock
    elif stock <= OVERSTOCK_LEVEL:
        return 1  # Medium Stock
    else:
        return 2  # High Stock

# === 🔹 Streamlit UI ===
st.title("📦 Smart Inventory Management System")
st.subheader("🔍 Real-Time IoT Monitoring & AI-Powered Insights")

# === 📡 Real-Time IoT Inventory Monitoring ===
st.subheader("📡 Live IoT Inventory Data")
latest_data = fetch_latest_data()
if not latest_data.empty:
    st.dataframe(latest_data)

    # 📊 Stock Levels Visualization
    fig_stock_levels = px.bar(
        latest_data, x='product_key', y='current_stock',
        color='product_key', title="📊 Current Stock Levels",
        labels={'product_key': "Product ID", 'current_stock': "Stock Level"}
    )
    st.plotly_chart(fig_stock_levels, use_container_width=True)

    # ⚠️ Alerts for Low Stock & Overstock
    low_stock = latest_data[latest_data["current_stock"] < latest_data["reorder_level"]]
    overstock = latest_data[latest_data["current_stock"] > 120]

    if not low_stock.empty:
        message = f"⚠️ Low Stock Alert: {len(low_stock)} items need restocking!"
        st.warning(message)
        send_telegram_message(message)
        send_email("Inventory Alert", message, "vijayabalan30042001@gmail.com")

    if not overstock.empty:
        message = f"🚨 Overstock Warning: {len(overstock)} items have excess inventory!"
        st.error(message)
        send_telegram_message(message)
        send_email("Inventory Alert", message, "vijayabalan30042001@gmail.com")

# === 📈 Sales Forecasting (ARIMA) ===
st.subheader("📈 Sales Forecasting")
uploaded_sales = st.file_uploader("Upload Sales Data (CSV)", type=["csv"])
if uploaded_sales:
    sales_df = pd.read_csv(uploaded_sales, parse_dates=["Order_Date"])
    sales_agg = sales_df.groupby("Order_Date")["Quantity"].sum().reset_index()
    sales_agg.set_index("Order_Date", inplace=True)

    predictions_arima = arima_model.forecast(steps=30)
    future_dates = pd.date_range(start=sales_agg.index[-1] + pd.Timedelta(days=1), periods=30, freq="D")
    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Sales": predictions_arima})

    st.dataframe(forecast_df)

    fig_sales_forecast = px.line(
        x=list(sales_agg.index) + list(future_dates),
        y=list(sales_agg["Quantity"]) + list(predictions_arima),
        labels={"x": "Date", "y": "Sales Quantity"},
        title="📈 Actual vs Forecasted Sales",
        line_shape="linear"
    )
    st.plotly_chart(fig_sales_forecast, use_container_width=True)

# === ⚠️ Anomaly Detection (Isolation Forest) ===
st.subheader("⚠️ Anomaly Detection")
uploaded_inventory = st.file_uploader("Upload Inventory Data (CSV)", type=["csv"])
if uploaded_inventory:
    inventory_df = pd.read_csv(uploaded_inventory)
    required_features = ["Initial_Stock", "Current_Stock", "Reorder_Level"]

    if all(feature in inventory_df.columns for feature in required_features):
        inventory_df["Anomaly"] = iso_forest.predict(inventory_df[required_features])
        anomalies = inventory_df[inventory_df["Anomaly"] == -1]

        st.write(anomalies if not anomalies.empty else "✅ No anomalies detected.")

        fig_anomalies = px.scatter(
            inventory_df, x="Product_Key", y="Current_Stock",
            color=inventory_df["Anomaly"].map({1: "Normal", -1: "Anomaly"}),
            title="⚠️ Inventory Anomaly Detection"
        )
        st.plotly_chart(fig_anomalies, use_container_width=True)
    else:
        st.error("Uploaded file is missing required columns: Initial_Stock, Current_Stock, Reorder_Level")

# === 🤖 AI Restocking Suggestions ===
st.subheader("🤖 AI Restocking Suggestions")
selected_product = st.selectbox("Select a Product:", collab_recommendations['Base_Product_Key'].unique())
current_stock = st.number_input("Enter Current Stock Level:", min_value=0, step=1)

if st.button("Get AI Restocking Suggestion"):
    product_state = classify_state(current_stock)
    best_action = np.argmax(q_table[product_state])
    st.success(f"🔹 Best Restocking Action: {ACTIONS[best_action]}")

    # Collaborative Filtering Recommendations
    st.subheader("📌 Alternative High-Demand Products")
    collab_results = collab_recommendations[collab_recommendations['Base_Product_Key'] == selected_product]
    if not collab_results.empty:
        st.write(collab_results[['Product_Key', 'Similarity_Score']])
    else:
        st.warning("No recommendations found for this product.")

    # Content-Based Recommendations
    st.subheader("🔄 Similar Products")
    content_results = content_recommendations[content_recommendations['Source_Product_Key'] == selected_product]
    if not content_results.empty:
        st.write(content_results[['Product_Key', 'Product_Name']])
    else:
        st.warning("No similar products found.")

    # Plot Q-Learning Decision Map
    st.subheader("📊 Stock-Level Impact Visualization")
    stock_levels = ["Low Stock", "Medium Stock", "High Stock"]
    q_values = q_table[product_state]
    fig = px.bar(
        x=stock_levels, y=q_values,
        labels={'x': 'Stock State', 'y': 'Q-Value'},
        title='Q-Learning Decision Map'
    )
    st.plotly_chart(fig)


CHATBOT.PY
import streamlit as st
import mysql.connector
import joblib
import pandas as pd
import numpy as np

# Load Machine Learning Models
q_table = joblib.load("C:/Users/vijayabalan/Downloads/optimized_q_table.pkl")
iso_forest = joblib.load("C:/Users/vijayabalan/Downloads/isolation_forest.pkl")
arima_model = joblib.load("C:/Users/vijayabalan/Downloads/arima_model.pkl")
collab_recommendations = pd.read_csv("C:/Users/vijayabalan/Downloads/Collaborative_Filtering_Recommendations_Fixed.csv")
content_recommendations = pd.read_csv("C:/Users/vijayabalan/Downloads/Content_Based_Recommendations_Fixed.csv")

# Constants for Stock Classification
REORDER_LEVEL = 15
OVERSTOCK_LEVEL = 40
ACTIONS = ["No Restock", "Medium Restock", "Large Restock"]

# Function to classify stock level
def classify_state(stock):
    if stock <= REORDER_LEVEL:
        return 0  # Low Stock
    elif stock <= OVERSTOCK_LEVEL:
        return 1  # Medium Stock
    else:
        return 2  # High Stock

# MySQL Database Connection
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="VIJAY@25",
        database="SmartInventory"
    )

# Chatbot responses
def get_response(user_input):
    responses = {
        "hello": "Hello! How can I assist you with inventory management today?",
        "stock levels": "You can check current stock levels by entering the product ID.",
        "reorder": "The system suggests reordering when stock falls below the reorder level.",
        "forecast": "Sales forecasting helps predict future demand using AI models.",
        "anomalies": "Anomalies in inventory are detected using AI to identify unusual stock patterns.",
        "thank you": "You're welcome! Let me know if you need further assistance."
    }
    for key in responses:
        if key in user_input.lower():
            return responses[key]
    return "I'm not sure about that. Can you ask something related to inventory?"

# Streamlit UI
st.title("🤖 Inventory Chatbot")
st.write("Ask me anything about inventory management!")

# User input
user_input = st.text_input("You:", "")

if user_input:
    if "restock" in user_input.lower():
        product_id = st.text_input("Enter Product ID:", "")
        if product_id:
            db = get_connection()
            cursor = db.cursor(dictionary=True)
            cursor.execute("SELECT * FROM InventoryData WHERE product_key = %s ORDER BY timestamp DESC LIMIT 1", (product_id,))
            product_data = cursor.fetchone()
            db.close()

            if product_data:
                current_stock = product_data["current_stock"]
                state = classify_state(current_stock)
                best_action = np.argmax(q_table[state])
                st.success(f"🔹 Best Restocking Action: {ACTIONS[best_action]}")
            else:
                st.warning("Product ID not found in inventory.")

    elif "anomaly" in user_input.lower():
        st.write("🔍 Checking for anomalies in inventory...")
        db = get_connection()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT product_key, current_stock, reorder_level FROM InventoryData")
        inventory_data = cursor.fetchall()
        db.close()

        # Convert to DataFrame
        df = pd.DataFrame(inventory_data)

        # Ensure proper feature alignment for Isolation Forest
        df["Anomaly"] = iso_forest.predict(df[["product_key", "current_stock", "reorder_level"]])

        # Display anomalies
        anomalies = df[df["Anomaly"] == -1]
        if not anomalies.empty:
            st.warning(f"⚠️ {len(anomalies)} anomalies detected in inventory!")
            st.dataframe(anomalies)
        else:
            st.success("✅ No anomalies detected.")

    elif "forecast" in user_input.lower():
        st.write("📈 Generating sales forecast...")
        predictions_arima = arima_model.forecast(steps=30)
        future_dates = pd.date_range(start=pd.Timestamp.today(), periods=30, freq="D")
        forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Sales": predictions_arima})
        st.dataframe(forecast_df)

    elif "recommend" in user_input.lower():
        product_id = st.text_input("Enter Product ID:", "")
        if product_id:
            st.subheader("📌 Alternative High-Demand Products")
            if product_id in collab_recommendations['Base_Product_Key'].values:
                collab_results = collab_recommendations[collab_recommendations['Base_Product_Key'] == product_id]
                if not collab_results.empty:
                    st.write(collab_results[['Product_Key', 'Similarity_Score']])
                else:
                    st.warning("No recommendations found for this product.")
            else:
                st.warning("No collaborative recommendations available.")

            st.subheader("🔄 Similar Products")
            if product_id in content_recommendations['Source_Product_Key'].values:
                content_results = content_recommendations[content_recommendations['Source_Product_Key'] == product_id]
                if not content_results.empty:
                    st.write(content_results[['Product_Key', 'Product_Name']])
                else:
                    st.warning("No similar products found.")
            else:
                st.warning("No content-based recommendations available.")

    else:
        response = get_response(user_input)
        st.write(f"🤖 Chatbot: {response}")
