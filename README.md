# Smart Inventory Management System

## 📌 Overview
The **Smart Inventory Management System** is a real-time, AI-powered solution for efficient stock management. It integrates **Machine Learning (ML), Internet of Things (IoT), and Data Analytics** to optimize inventory levels, detect anomalies, and automate restocking decisions.

## 🚀 Features
### 🔹 Real-Time IoT Monitoring
- Uses MQTT protocol for real-time inventory updates
- Simulates IoT data for stock levels

### 🔹 AI-Powered Inventory Forecasting
- **Demand Prediction:** Facebook Prophet for sales forecasting
- **Anomaly Detection:** Isolation Forest for outlier detection
- **Restocking Optimization:** Q-learning-based reinforcement learning
- **Recommendation System:** Collaborative & Content-based filtering for restocking

### 🔹 Smart Alerts & Notifications
- **Telegram Bot** integration for real-time inventory alerts
- **Email Notifications** for low stock alerts

### 🔹 Interactive Dashboard (Streamlit)
- Visualizes inventory trends using **Plotly & Matplotlib**
- AI-driven recommendations for restocking decisions
- Chatbot for inventory queries using OpenAI API

## 🏗️ Tech Stack
- **Backend:** Python, Flask, MySQL
- **Frontend:** Streamlit, Plotly, Shadcn/UI
- **Machine Learning:** Scikit-learn, Facebook Prophet, Reinforcement Learning
- **IoT Simulation:** MQTT (Paho-MQTT)
- **Notification System:** Telegram API, SMTP (Email)

## 📁 Project Structure
```
📂 smart-inventory-management
├── 📁 data                # Sample datasets (Stores, Sales, Products, Inventory)
├── 📁 models              # Saved ML models (Q-learning, Prophet, Isolation Forest)
├── 📁 scripts             # Python scripts for ML training & IoT simulation
├── 📁 streamlit_app       # Streamlit dashboard implementation
├── 📄 requirements.txt    # Python dependencies
├── 📄 README.md           # Project documentation
└── 📄 app.py              # Main Streamlit application file
```

## 🔧 Installation & Setup
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/your-username/smart-inventory-management.git
cd smart-inventory-management
```

### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 3️⃣ Set Up Environment Variables
Create a `.env` file to store credentials securely:
```
MYSQL_HOST=your_mysql_host
MYSQL_USER=your_mysql_user
MYSQL_PASSWORD=your_mysql_password
TELEGRAM_API_TOKEN=your_telegram_bot_token
EMAIL_PASSWORD=your_email_password
```

### 4️⃣ Run the Streamlit App
```sh
streamlit run app.py
```

## 📊 Visualizations & Dashboards
- **Real-time inventory monitoring** with IoT simulated data
- **AI-based demand forecasting** using Prophet
- **Anomaly detection** for stock irregularities
- **Smart restocking recommendations** with Reinforcement Learning

## 🤖 Future Enhancements
- **Real IoT Hardware Integration** with sensors
- **Automated Purchase Order Generation**
- **Integration with ERP Systems**

## 🤝 Contributing
Contributions are welcome! Feel free to fork this repository and submit a pull request.

## 📜 License
This project is licensed under the MIT License.

---
📌 **Developed by vijayabalan** 🚀

