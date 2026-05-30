---
title: ChargePath India EV EDA Dashboard
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
---

# ⚡ ChargePath India — EV Charging Network EDA Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

> **MBA Data Analytics · Individual Assignment**  
> Task 1: Synthetic Data (10M) | Task 2: Data Cleaning (10M) | Task 3: EDA & Charts (30M)

---

## 🚀 Live Demo

| Platform | Link |
|----------|------|
| **Streamlit Cloud** | [Deploy instructions below](#deploy-on-streamlit-cloud) |
| **HuggingFace Spaces** | [Deploy instructions below](#deploy-on-huggingface-spaces) |

---

## 📋 Project Overview

**ChargePath India** is a synthetic EV public charging station start-up dataset built to validate a real business idea:  
_Can a network of EV charging stations across Indian cities generate profitable unit economics within 6–8 months?_

### Dataset
- **200 stations** generated → **197 after cleaning** (3 duplicates removed)
- **25 variables**: 17 raw + 8 derived features
- **8 Indian states**: Maharashtra, Karnataka, Delhi, Tamil Nadu, Gujarat, Telangana, Rajasthan, West Bengal
- **3 charger types**: DC Fast Charger, AC Level 2, Ultra-Fast DC
- **6 station types**: Highway, Mall, Office Park, Residential, Airport, Hotel

---

## 📊 Dashboard Features

| Tab | Content |
|-----|---------|
| **Overview & KPIs** | 8 live KPI cards, geographic distribution charts |
| **Data Cleaning** | Missing values viz, outlier detection, cleaning log, feature engineering |
| **Descriptive Stats** | Full summary statistics table + frequency distributions |
| **EDA Charts** | 12 analytical charts with business insights |
| **Correlation** | Pearson correlation heatmap + key findings |
| **Dataset** | Searchable data explorer with CSV download |

### Charts Included
1. Average Revenue by Station Type (Horizontal Bar)
2. Average Revenue by State (Bar + Network Average line)
3. Sessions Per Day Distribution (Histogram)
4. Profit by Region (Bar chart)
5. Charger Type Distribution (Pie chart)
6. Revenue Category Breakdown (Bar)
7. Customer Rating by Station Type
8. Profit Margin % by Charger Type (Box Plot)
9. Revenue vs Sessions/Day (Scatter + Trend)
10. Uptime % vs Customer Rating (Scatter + Correlation)
11. Revenue by Installation Year (Line + Bar combo)
12. Profitable vs Loss-Making Stations (Donut)
+ Pearson Correlation Heatmap (10×10)

---

## 🛠️ Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/chargepath-eda.git
cd chargepath-eda

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 🌐 Deploy on Streamlit Cloud

> Free deployment in 3 steps — no credit card needed

**Step 1:** Push this repo to GitHub (public repo)
```bash
git init
git add .
git commit -m "Initial commit: ChargePath India EDA Dashboard"
git remote add origin https://github.com/YOUR_USERNAME/chargepath-eda.git
git push -u origin main
```

**Step 2:** Go to [share.streamlit.io](https://share.streamlit.io)  
- Click **"New app"**
- Select your GitHub repo: `YOUR_USERNAME/chargepath-eda`
- Branch: `main`
- Main file path: `app.py`
- Click **"Deploy"**

**Step 3:** Your app is live at:  
`https://YOUR_USERNAME-chargepath-eda-app-XXXXX.streamlit.app`

---

## 🤗 Deploy on HuggingFace Spaces

> The `README.md` front-matter is already configured for HuggingFace Spaces

**Step 1:** Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
- Name: `chargepath-eda`
- SDK: **Streamlit**
- Visibility: Public

**Step 2:** Upload files via the web UI or Git:
```bash
# Using HuggingFace CLI
pip install huggingface_hub

huggingface-cli login
# Paste your HF token from huggingface.co/settings/tokens

# Clone your space
git clone https://huggingface.co/spaces/YOUR_HF_USERNAME/chargepath-eda
cd chargepath-eda

# Copy app files
cp /path/to/app.py .
cp /path/to/requirements.txt .
cp /path/to/README.md .

# Push
git add .
git commit -m "Deploy ChargePath EDA Dashboard"
git push
```

**Step 3:** Your app is live at:  
`https://YOUR_HF_USERNAME-chargepath-eda.hf.space`

---

## 📁 File Structure

```
chargepath-eda/
├── app.py              ← Main Streamlit app (all-in-one, no external data files needed)
├── requirements.txt    ← Python dependencies
└── README.md           ← This file (also HuggingFace Spaces config)
```

> **Note:** The dataset is fully generated inside `app.py` using Python's `random` module with a fixed seed (`random.seed(42)`), so **no CSV file is needed**. The app is fully self-contained.

---

## 🔧 Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.9+ | Core language |
| Streamlit | Web app framework |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Chart rendering |
| Seaborn | Heatmap & statistical plots |

---

## 📚 Assignment Tasks Covered

| Task | Marks | Description |
|------|-------|-------------|
| **Task 1** | 10M | Synthetic data generation for EV charging business validation |
| **Task 2** | 10M | Data cleaning (missing values, outliers, duplicates) + transformation (8 derived features) |
| **Task 3** | 30M | Descriptive analytics + 12 EDA charts + correlation matrix with business insights |
| **Total** | **50M** | Individual submission (Session 5) |

---

## 📄 License

MIT License — free to use, modify, and distribute.
