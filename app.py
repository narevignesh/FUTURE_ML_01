import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import kagglehub
import os

# Page Config
st.set_page_config(page_title="Big Mart Sales Dashboard", layout="wide", page_icon="🏢")

# Custom Styling
st.markdown("""
    <style>
        .main { background-color: #f7f9fc; font-family: 'Segoe UI', sans-serif; }
        h1, h2, h3, h4, h5, h6 { color: #00274d; font-family: 'Trebuchet MS', sans-serif; }
        .stButton>button { background-color: #004080; color: white; border-radius: 5px; }
        .stButton>button:hover { background-color: #0066cc; }
        .reportbox {
            background-color: #f0f8ff;
            padding: 20px;
            border-radius: 12px;
            margin-top: 40px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
            font-size: 16px;
            line-height: 1.6;
        }
        .footer {
            text-align: center;
            color: grey;
            margin-top: 50px;
            font-size: 14px;
        }
        hr { border: 1px solid #cccccc; }
    </style>
""", unsafe_allow_html=True)

# Download the dataset
mragpavank_big_mart_sales_dataset_path = kagglehub.dataset_download('mragpavank/big-mart-sales-dataset')
bigmart_train = pd.read_csv(os.path.join(mragpavank_big_mart_sales_dataset_path, 'Train.csv'))

# Make a copy for raw visualization before transformations
bigmart_train_raw = bigmart_train.copy()

# Data Preprocessing
bigmart_train['Outlet_Size'] = bigmart_train['Outlet_Size'].map({"Small": 1, "Medium": 2, "High": 3})
bigmart_train['Item_Weight'] = bigmart_train['Item_Weight'].fillna(bigmart_train['Item_Weight'].median())
bigmart_train['Outlet_Size'].fillna(bigmart_train['Outlet_Size'].mode()[0], inplace=True)
bigmart_train.drop(labels=["Outlet_Establishment_Year"], axis=1, inplace=True)

# Encoding Categorical Features
feat = ['Outlet_Size', 'Outlet_Type', 'Outlet_Location_Type', 'Item_Fat_Content', "Item_Type"]
X = pd.get_dummies(bigmart_train[feat])
bigmart_train = pd.concat([bigmart_train, X], axis=1)

# Drop old categorical columns and identifiers
bigmart_train.drop(labels=["Outlet_Size", 'Outlet_Location_Type', "Outlet_Type", 'Item_Fat_Content',
                          'Outlet_Identifier', 'Item_Identifier', "Item_Type"], axis=1, inplace=True)

# Feature Scaling
X_train_prescale = bigmart_train.drop(labels=["Item_Outlet_Sales"], axis=1)
y_train = bigmart_train["Item_Outlet_Sales"]
scaler = preprocessing.MinMaxScaler()
X_scaled = scaler.fit_transform(X_train_prescale)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_train, test_size=0.4, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)

# Title Section
st.markdown("""
<div style='background-color: #cce7ff; padding: 18px; border-radius: 12px; margin-bottom: 20px;'>
    <h1 style='text-align: center; color: #003366;'>🏢 Big Mart Sales Interactive Dashboard</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<h3 style='color:#003366;'>📅 Overview</h3>
<p style='font-size:16px;'>Gain insights into Big Mart's top-selling products and trends! Explore item sales across different MRP levels, fat content, outlet types, and more. This dashboard enables predictive analytics for future product strategies.</p>
""", unsafe_allow_html=True)

# Visualizations
st.markdown("## 📊 Visualizations", unsafe_allow_html=True)

with st.expander("📂 View Raw Dataset"):
    st.dataframe(bigmart_train_raw.head(20), use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Sales Distribution 📦")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(bigmart_train["Item_Outlet_Sales"], bins=50, kde=True, color='deepskyblue', ax=ax)
    ax.set_xlabel("Sales", fontsize=10)
    st.pyplot(fig)

    st.subheader("Sales vs MRP Scatter Plot 🛍️")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(x='Item_MRP', y='Item_Outlet_Sales', data=bigmart_train, hue='Item_Weight', palette='viridis', ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("MRP Distribution 💰")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(bigmart_train["Item_MRP"], bins=60, kde=True, color='coral', ax=ax)
    ax.set_xlabel("MRP", fontsize=10)
    st.pyplot(fig)

    st.subheader("Item Visibility vs Sales 📢")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(x='Item_Visibility', y='Item_Outlet_Sales', data=bigmart_train, color='hotpink', ax=ax)
    st.pyplot(fig)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Boxplot of Sales based on Outlet Size 📦")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(x='Outlet_Size', y='Item_Outlet_Sales', data=bigmart_train_raw, palette='pastel', ax=ax)
    st.pyplot(fig)

with col4:
    st.subheader("Sales by Fat Content (Pie Chart) 🥧")
    fat_content_counts = bigmart_train_raw['Item_Fat_Content'].value_counts()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(fat_content_counts, labels=fat_content_counts.index, autopct='%1.1f%%', colors=sns.color_palette('Set2'))
    st.pyplot(fig)

st.subheader("Residual Plot (Model Errors) 🛠️")
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot((y_test - preds), bins=50, kde=True, color='limegreen', ax=ax)
st.pyplot(fig)

# Report Below Visualization
st.markdown("""
<div class="reportbox">
    <h3>📄 Sales Performance Report</h3>
    <ul>
        <li><strong>Top Products Sell More When MRP is Higher:</strong> Items with MRP above ₹150 show significantly better sales.</li>
        <li><strong>Outlet Size Matters:</strong> Medium and High outlet sizes show consistent better performance.</li>
        <li><strong>Fat Content Trends:</strong> Low Fat and Regular items dominate the market with minimal influence on overall sales.</li>
        <li><strong>Visibility Factor:</strong> Items placed more visibly are not always better sellers; quality and MRP seem to matter more.</li>
        <li><strong>Sales Distribution:</strong> Most items sell under ₹4000 but select items exceed ₹6000+.</li>
        <li><strong>Boxplot Insight:</strong> Certain outlets consistently sell more, likely due to better footfall or larger inventory.</li>
    </ul>
    <p><em>Use this report to make smarter stocking, pricing, and marketing decisions for your Big Mart branches.</em></p>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr>
    <div class="footer">
        Made by Vignesh using Streamlit | Big Mart Sales 2025
    </div>
""", unsafe_allow_html=True)
