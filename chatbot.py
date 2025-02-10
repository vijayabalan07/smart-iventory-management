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
