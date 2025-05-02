# 🏢 Big Mart Sales Interactive Dashboard

A beautiful and insightful sales analysis dashboard built with **Streamlit** and **scikit-learn**. It uses the Big Mart dataset and applies linear regression to predict item outlet sales while offering various visualizations to explore trends and patterns.



## 🚀 Features

- 📦 Preprocessed Big Mart dataset with missing values handled  
- 📊 Exploratory Data Analysis with Seaborn and Matplotlib  
- 🤖 Linear Regression Model for sales prediction  
- 📈 Beautiful plots and interactive widgets  
- 🧠 Powered by `scikit-learn`, `streamlit`, `seaborn`, and `matplotlib`  
- 🔍 Visual insights into MRP, Outlet Type, Size, Fat Content, and Sales distribution  

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/bigmart-sales-dashboard.git
cd bigmart-sales-dashboard
```

### 2. Install Required Libraries
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset from Kaggle
This project uses the Big Mart dataset hosted on Kaggle.

Ensure you have:
- A Kaggle account
- Your `kaggle.json` file placed in `~/.kaggle/` directory

Or let `kagglehub` download it for you from:
```
https://www.kaggle.com/datasets/mragpavank/big-mart-sales-dataset
```

> The app handles automatic download using [`kagglehub`](https://github.com/ayulockin/kagglehub).

---

### 4. Run the Dashboard

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📂 Project Structure

```
bigmart-sales-dashboard/
│
├── app.py                # Main Streamlit app
├── requirements.txt      # Required packages
└── README.md             # Documentation
```

---

## 🙋‍♂️ Author

Made with 💙 by [Vignesh](https://github.com/narevignesh)  
Big Mart Sales Insights • 2025

---
