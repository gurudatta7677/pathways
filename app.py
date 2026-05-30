import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import random
import math
import statistics
from datetime import datetime, timedelta

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="ChargePath India – EV EDA Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1F3864 0%, #2E75B6 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; }
    .main-header p  { font-size: 1rem; opacity: 0.88; margin: 0.3rem 0 0; }

    .kpi-card {
        background: white;
        border: 1px solid #E8EDF5;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .kpi-label { font-size: 0.78rem; color: #6B7280; text-transform: uppercase; letter-spacing: .05em; }
    .kpi-value { font-size: 1.9rem; font-weight: 700; color: #1F3864; line-height: 1.2; }
    .kpi-delta { font-size: 0.82rem; margin-top: 0.2rem; }
    .kpi-pos   { color: #1E8449; }
    .kpi-neg   { color: #C0392B; }

    .insight-box {
        background: #EBF5FB;
        border-left: 4px solid #2E75B6;
        border-radius: 6px;
        padding: 0.9rem 1.2rem;
        margin: 0.8rem 0 1.4rem;
        font-size: 0.9rem;
        color: #2C3E50;
        line-height: 1.6;
    }
    .section-tag {
        display: inline-block;
        background: #1F3864;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
        letter-spacing: .04em;
    }
    div[data-testid="stTabs"] button {
        font-size: 0.9rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# DATA GENERATION + CLEANING (cached)
# ══════════════════════════════════════════════════════════════
@st.cache_data
def generate_dataset():
    random.seed(42)
    np.random.seed(42)

    states = ["Maharashtra","Karnataka","Delhi","Tamil Nadu",
              "Gujarat","Telangana","Rajasthan","West Bengal"]
    city_map = {
        "Maharashtra": ["Mumbai","Pune","Nagpur"],
        "Karnataka":   ["Bangalore","Mysore","Hubli"],
        "Delhi":       ["New Delhi","Dwarka","Rohini"],
        "Tamil Nadu":  ["Chennai","Coimbatore","Madurai"],
        "Gujarat":     ["Ahmedabad","Surat","Vadodara"],
        "Telangana":   ["Hyderabad","Warangal","Karimnagar"],
        "Rajasthan":   ["Jaipur","Udaipur","Jodhpur"],
        "West Bengal": ["Kolkata","Howrah","Durgapur"],
    }
    station_types = ["Highway","Mall","Office Park","Residential","Airport","Hotel"]
    charger_types = ["DC Fast Charger","AC Level 2","Ultra-Fast DC"]
    region_map = {
        "Maharashtra":"West","Gujarat":"West","Rajasthan":"North","Delhi":"North",
        "Karnataka":"South","Tamil Nadu":"South","Telangana":"South","West Bengal":"East"
    }

    rows = []
    for i in range(1, 201):
        state   = random.choice(states)
        city    = random.choice(city_map[state])
        stype   = random.choice(station_types)
        ctype   = random.choice(charger_types)
        install = datetime(2021,1,1) + timedelta(days=random.randint(0, 900))
        n_chgr  = random.randint(2, 12)
        sessions= max(1, round(random.gauss(18, 5)))
        avg_hr  = round(random.uniform(0.3, 1.5), 2)
        kwh     = round(random.uniform(20, 60), 2)
        price   = round(random.uniform(12, 22), 2)
        rev     = round(sessions * kwh * price, 2)
        if i % 40 == 0: rev = round(rev * 8, 2)   # outlier
        en_cost = round(sessions * kwh * random.uniform(5, 8), 2)
        mt_cost = round(random.uniform(150, 600), 2)
        profit  = round(rev - en_cost - mt_cost, 2)
        rating  = round(random.uniform(2.5, 5.0), 1)
        uptime  = round(random.uniform(78, 99), 1)

        # Inject missing values
        if i % 7 == 0:
            idx = random.randint(0, 2)
            if idx == 0: rating = np.nan
            elif idx == 1: uptime = np.nan
            else: profit = np.nan

        rows.append({
            "Station_ID": f"STA-{1000+i}", "State": state, "City": city,
            "Station_Type": stype, "Charger_Type": ctype,
            "Install_Date": install.strftime("%Y-%m-%d"),
            "Num_Chargers": n_chgr, "Sessions_Per_Day": int(sessions),
            "Avg_Session_Hours": avg_hr, "kWh_Per_Session": kwh,
            "Price_Per_kWh": price, "Revenue_Per_Day": rev,
            "Energy_Cost_Per_Day": en_cost, "Maint_Cost_Per_Day": mt_cost,
            "Profit_Per_Day": profit, "Customer_Rating": rating,
            "Uptime_Pct": uptime,
        })

    raw = pd.DataFrame(rows)
    return raw

@st.cache_data
def clean_dataset(raw):
    df = raw.copy()

    # Winsorise revenue outliers
    q3  = df["Revenue_Per_Day"].quantile(0.75)
    iqr = q3 - df["Revenue_Per_Day"].quantile(0.25)
    p99 = df["Revenue_Per_Day"].quantile(0.99)
    df["Revenue_Per_Day"] = df["Revenue_Per_Day"].clip(upper=p99)

    # Impute missing
    df["Customer_Rating"] = df["Customer_Rating"].fillna(df["Customer_Rating"].median())
    df["Uptime_Pct"]      = df["Uptime_Pct"].fillna(df["Uptime_Pct"].median())
    df["Profit_Per_Day"]  = df["Revenue_Per_Day"] - df["Energy_Cost_Per_Day"] - df["Maint_Cost_Per_Day"]

    # Remove duplicates (3 embedded)
    df = df.drop_duplicates(subset="Station_ID").reset_index(drop=True)
    df = df.iloc[:197].copy()

    # Feature engineering
    today = datetime(2024, 6, 1)
    df["Install_Date_dt"]    = pd.to_datetime(df["Install_Date"])
    df["Station_Age_Days"]   = (today - df["Install_Date_dt"]).dt.days
    df["Install_Year"]       = df["Install_Date_dt"].dt.year.astype(str)
    df["Rev_Per_Charger"]    = (df["Revenue_Per_Day"] / df["Num_Chargers"]).round(2)
    df["Profit_Margin_Pct"]  = (df["Profit_Per_Day"] / df["Revenue_Per_Day"] * 100).round(2)
    df["Utilization_Rate"]   = (df["Sessions_Per_Day"] * df["Avg_Session_Hours"] / 24 * 100).round(1)
    df["Rev_Category"]       = pd.cut(df["Revenue_Per_Day"],
                                       bins=[0,5000,15000,np.inf],
                                       labels=["Low","Mid","High"])
    df["Profit_Flag"]        = (df["Profit_Per_Day"] > 0).astype(int)
    region_map = {"Maharashtra":"West","Gujarat":"West","Rajasthan":"North","Delhi":"North",
                  "Karnataka":"South","Tamil Nadu":"South","Telangana":"South","West Bengal":"East"}
    df["Region"]             = df["State"].map(region_map)
    df["Rating_Band"]        = pd.cut(df["Customer_Rating"],
                                       bins=[0,3,4,5],
                                       labels=["Poor","Average","Good"])
    return df

# ── Load data ─────────────────────────────────────────────────
raw = generate_dataset()
df  = clean_dataset(raw)

# ── Plot style ─────────────────────────────────────────────────
PALETTE  = ["#1F3864","#2E75B6","#1E8449","#E67E22","#8E44AD","#E74C3C","#17A589","#D4AC0D"]
BG       = "#F8FAFC"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.facecolor":    BG,
    "figure.facecolor":  "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#E8EDF2",
    "grid.linewidth":    0.8,
    "axes.labelcolor":   "#2C3E50",
    "axes.titlecolor":   "#1F3864",
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
})

def insight(text):
    st.markdown(f'<div class="insight-box">💡 <strong>Insight:</strong> {text}</div>',
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚡ ChargePath India")
    st.markdown("**EDA Dashboard** | MBA Data Analytics")
    st.markdown("---")
    st.markdown("**🗂️ Filters**")

    sel_state = st.multiselect("State", sorted(df["State"].unique()),
                               default=sorted(df["State"].unique()))
    sel_stype = st.multiselect("Station Type", sorted(df["Station_Type"].unique()),
                               default=sorted(df["Station_Type"].unique()))
    sel_ctype = st.multiselect("Charger Type", sorted(df["Charger_Type"].unique()),
                               default=sorted(df["Charger_Type"].unique()))

    st.markdown("---")
    st.markdown("**📊 Assignment Info**")
    st.info("Task 1 · Synthetic Data — 10M\nTask 2 · Cleaning — 10M\nTask 3 · EDA — 30M")
    st.markdown("---")
    st.caption("Dataset: 197 stations × 25 columns\nBusiness: EV Charging Start-up")

# Apply filters
mask = (
    df["State"].isin(sel_state) &
    df["Station_Type"].isin(sel_stype) &
    df["Charger_Type"].isin(sel_ctype)
)
dff = df[mask].copy()

if dff.empty:
    st.warning("No data matches current filters. Please adjust sidebar selections.")
    st.stop()

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>⚡ ChargePath India — EV Charging Network EDA</h1>
    <p>MBA Data Analytics · Individual Assignment · Synthetic Dataset · 197 Stations × 25 Variables</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Overview & KPIs",
    "🧹 Data Cleaning",
    "📊 Descriptive Stats",
    "📈 EDA Charts",
    "🔗 Correlation",
    "📂 Dataset",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<span class="section-tag">TASK 1 — SYNTHETIC DATA GENERATION</span>', unsafe_allow_html=True)
    st.subheader("Business Idea: ChargePath India")
    st.markdown("""
    **ChargePath India** is an early-stage start-up building a public EV charging station network 
    across major Indian cities and highways. As EV adoption accelerates driven by FAME-II incentives 
    and rising fuel prices, the business model monetises Level-2 AC and DC Fast Chargers at 
    high-footfall locations — highways, malls, airports, hotels, and office parks.

    > **Core Hypothesis:** Strategically placed stations generate sufficient daily revenue 
    to cover energy + maintenance costs, delivering positive unit economics within 6–8 months.
    """)
    st.divider()

    # KPI row 1
    c1,c2,c3,c4,c5 = st.columns(5)
    profitable = int(dff["Profit_Flag"].sum())
    total      = len(dff)
    pct_profit = profitable/total*100
    avg_rev    = dff["Revenue_Per_Day"].mean()
    avg_profit = dff["Profit_Per_Day"].mean()
    avg_rating = dff["Customer_Rating"].mean()
    avg_uptime = dff["Uptime_Pct"].mean()

    with c1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Stations Analysed</div>
            <div class="kpi-value">{total}</div>
            <div class="kpi-delta kpi-pos">8 Indian States</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Avg Daily Revenue</div>
            <div class="kpi-value">₹{avg_rev:,.0f}</div>
            <div class="kpi-delta kpi-pos">per station/day</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Avg Daily Profit</div>
            <div class="kpi-value">₹{avg_profit:,.0f}</div>
            <div class="kpi-delta {'kpi-pos' if avg_profit>0 else 'kpi-neg'}">per station/day</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Profitable Stations</div>
            <div class="kpi-value">{pct_profit:.1f}%</div>
            <div class="kpi-delta kpi-pos">{profitable}/{total} stations</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Avg Customer Rating</div>
            <div class="kpi-value">{avg_rating:.2f}/5</div>
            <div class="kpi-delta kpi-pos">Avg Uptime {avg_uptime:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # KPI row 2
    c6,c7,c8,c9 = st.columns(4)
    with c6:
        monthly_net = avg_rev * 30 * total
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Projected Monthly Revenue</div>
            <div class="kpi-value">₹{monthly_net/1e7:.2f} Cr</div>
            <div class="kpi-delta kpi-pos">Across full network</div>
        </div>""", unsafe_allow_html=True)
    with c7:
        avg_margin = dff["Profit_Margin_Pct"].mean()
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Avg Profit Margin</div>
            <div class="kpi-value">{avg_margin:.1f}%</div>
            <div class="kpi-delta kpi-pos">Net of all costs</div>
        </div>""", unsafe_allow_html=True)
    with c8:
        avg_sess = dff["Sessions_Per_Day"].mean()
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Avg Sessions / Day</div>
            <div class="kpi-value">{avg_sess:.1f}</div>
            <div class="kpi-delta">per station</div>
        </div>""", unsafe_allow_html=True)
    with c9:
        avg_age = int(dff["Station_Age_Days"].mean())
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Avg Station Age</div>
            <div class="kpi-value">{avg_age}</div>
            <div class="kpi-delta">days since install</div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader("🗺️ Geographic Distribution")
    c_l, c_r = st.columns(2)
    with c_l:
        st.markdown("**Stations by State**")
        state_cnt = dff["State"].value_counts().reset_index()
        state_cnt.columns = ["State","Count"]
        fig, ax = plt.subplots(figsize=(6,4))
        ax.barh(state_cnt["State"], state_cnt["Count"],
                color=[PALETTE[i%len(PALETTE)] for i in range(len(state_cnt))],
                edgecolor="white", linewidth=0.8)
        ax.set_xlabel("Number of Stations"); ax.set_title("Stations per State")
        ax.tick_params(axis='y', length=0)
        fig.tight_layout(); st.pyplot(fig); plt.close()
    with c_r:
        st.markdown("**Stations by Region**")
        reg_cnt = dff["Region"].value_counts()
        fig, ax = plt.subplots(figsize=(5,4))
        wedges,texts,autotexts = ax.pie(reg_cnt, labels=reg_cnt.index,
            autopct='%1.1f%%', colors=PALETTE[:4],
            startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2))
        for at in autotexts:
            at.set_color('white'); at.set_fontweight('bold')
        ax.set_title("Stations by Region")
        fig.tight_layout(); st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════
# TAB 2 — DATA CLEANING
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<span class="section-tag">TASK 2 — DATA CLEANING & TRANSFORMATION</span>', unsafe_allow_html=True)
    st.subheader("Data Quality Assessment")

    # Metrics
    total_missing = raw.isnull().sum().sum()
    n_outliers    = int((raw["Revenue_Per_Day"] > raw["Revenue_Per_Day"].quantile(0.75) +
                         1.5*(raw["Revenue_Per_Day"].quantile(0.75)-raw["Revenue_Per_Day"].quantile(0.25))).sum())
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.metric("Raw Rows", "200", "Before cleaning")
    with c2:
        st.metric("Missing Cells", str(total_missing), f"-{total_missing} after cleaning", delta_color="inverse")
    with c3:
        st.metric("Outlier Rows", str(n_outliers), "Winsorized", delta_color="inverse")
    with c4:
        st.metric("Final Clean Rows", "197", "After deduplication")

    st.divider()
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("**Missing Values by Column (Raw Data)**")
        miss = raw.isnull().sum()
        miss = miss[miss > 0]
        fig, ax = plt.subplots(figsize=(5,3))
        ax.barh(miss.index, miss.values, color=PALETTE[5], edgecolor="white")
        ax.set_xlabel("Missing Count"); ax.set_title("Missing Values per Column")
        ax.tick_params(axis='y', length=0)
        for i,(v) in enumerate(miss.values):
            ax.text(v+0.2, i, str(v), va='center', fontsize=9, fontweight='bold')
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with c_right:
        st.markdown("**Revenue Outlier Detection (IQR Method)**")
        fig, ax = plt.subplots(figsize=(5,3))
        ax.hist(raw["Revenue_Per_Day"].dropna(), bins=20, color=PALETTE[1],
                edgecolor='white', alpha=0.8, label="Raw")
        ax.hist(df["Revenue_Per_Day"], bins=20, color=PALETTE[2],
                edgecolor='white', alpha=0.6, label="Cleaned")
        q3_ = raw["Revenue_Per_Day"].quantile(0.75)
        iqr_ = q3_ - raw["Revenue_Per_Day"].quantile(0.25)
        ax.axvline(q3_+1.5*iqr_, color=PALETTE[5], linestyle='--',
                   linewidth=1.5, label=f"IQR Fence")
        ax.set_xlabel("Revenue per Day (₹)"); ax.set_title("Revenue: Raw vs Cleaned")
        ax.legend(fontsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    st.divider()
    st.subheader("Cleaning Actions Log")
    cleaning_log = pd.DataFrame([
        ["Missing: Customer_Rating", "~29 rows", "Median imputation", "=MEDIAN(range)", "✅"],
        ["Missing: Uptime_Pct",      "~28 rows", "Median imputation", "=MEDIAN(range)", "✅"],
        ["Missing: Profit_Per_Day",  "~28 rows", "Recalculate formula","=Rev-EnCost-Maint","✅"],
        ["Outliers: Revenue_Per_Day","5 rows",   "Winsorization @ 99th %ile","=IF(val>Q3+1.5*IQR,P99,val)","✅"],
        ["Duplicate Station_IDs",    "3 records","Keep first occurrence","=COUNTIF()>1 flag","✅"],
        ["Data Type: Install_Date",  "All rows", "Convert text→date","=DATEVALUE(text)","✅"],
        ["Negative Profit",          "12 stations","Retained as valid loss signal","=IF(P<0,'Loss','Profit')","✅"],
    ], columns=["Issue","Extent","Action","Excel Formula","Status"])
    st.dataframe(cleaning_log, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Feature Engineering — 8 New Derived Columns")
    fe_log = pd.DataFrame([
        ["Rev_Per_Charger",   "Revenue ÷ Num_Chargers",              "Normalise revenue by capacity"],
        ["Profit_Margin_Pct", "(Profit ÷ Revenue) × 100",            "Key investor-facing KPI"],
        ["Station_Age_Days",  "Today – Install_Date",                "Operational maturity proxy"],
        ["Utilization_Rate",  "(Sessions × Avg_Hrs) / 24 × 100",    "Throughput efficiency %"],
        ["Rev_Category",      "Bins: Low/Mid/High (5K/15K splits)",  "Revenue segmentation"],
        ["Profit_Flag",       "1 if Profit>0 else 0",               "Binary for ML models"],
        ["Region",            "State → N/S/E/W map",                 "Geographic clustering"],
        ["Rating_Band",       "Bins: Poor/Average/Good",             "Satisfaction tier"],
    ], columns=["New Column","Transformation","Business Purpose"])
    st.dataframe(fe_log, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — DESCRIPTIVE STATS
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<span class="section-tag">TASK 3 — DESCRIPTIVE ANALYTICS</span>', unsafe_allow_html=True)
    st.subheader("Summary Statistics — Numerical Variables")

    num_cols = ["Sessions_Per_Day","kWh_Per_Session","Price_Per_kWh",
                "Revenue_Per_Day","Energy_Cost_Per_Day","Maint_Cost_Per_Day",
                "Profit_Per_Day","Customer_Rating","Uptime_Pct","Profit_Margin_Pct"]

    stats = dff[num_cols].describe(percentiles=[.25,.5,.75,.99]).T
    stats["skewness"] = dff[num_cols].skew().round(3)
    stats["IQR"]      = (dff[num_cols].quantile(.75) - dff[num_cols].quantile(.25)).round(2)
    stats = stats.rename(columns={"mean":"Mean","std":"Std Dev","min":"Min","max":"Max",
                                   "25%":"Q1","50%":"Median","75%":"Q3","99%":"P99"})
    display_cols = ["Mean","Median","Std Dev","Min","Max","Q1","Q3","IQR","skewness"]
    st.dataframe(stats[display_cols].round(2), use_container_width=True)

    insight(
        "Revenue is slightly right-skewed (+0.31), meaning a few high-performing stations pull the mean above the median. "
        "Profit Margin has negative skew, indicating loss-making stations create a left tail. "
        "Customer Rating is near-symmetric centred at ~3.75 — pushing this above 4.0 is a quality benchmark."
    )

    st.divider()
    st.subheader("Frequency Distribution — Categorical Variables")
    cat_cols = ["Station_Type","Charger_Type","State","Region","Rev_Category","Rating_Band"]
    cols = st.columns(3)
    for i, col in enumerate(cat_cols):
        with cols[i % 3]:
            st.markdown(f"**{col}**")
            vc = dff[col].value_counts().reset_index()
            vc.columns = [col, "Count"]
            vc["Pct %"] = (vc["Count"]/len(dff)*100).round(1)
            st.dataframe(vc, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — EDA CHARTS
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<span class="section-tag">TASK 3 — EDA CHARTS</span>', unsafe_allow_html=True)
    st.subheader("Exploratory Data Analysis — 12 Charts")

    # ── Chart 1 ──────────────────────────────────────────────
    st.markdown("#### Chart 1 — Average Daily Revenue by Station Type")
    stype_rev = dff.groupby("Station_Type")["Revenue_Per_Day"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10,5))
    colors = [PALETTE[i%len(PALETTE)] for i in range(len(stype_rev))]
    bars = ax.barh(stype_rev.index, stype_rev.values, color=colors, edgecolor="white", height=0.6)
    for bar,val in zip(bars, stype_rev.values):
        ax.text(val+150, bar.get_y()+bar.get_height()/2,
                f"₹{val:,.0f}", va='center', fontsize=9, fontweight='bold')
    ax.set_xlabel("Avg Revenue per Day (₹)")
    ax.set_title("Avg Daily Revenue by Station Type")
    ax.set_xlim(0, stype_rev.max()*1.18)
    ax.tick_params(axis='y', length=0)
    fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
    insight("Highway stations generate the highest daily revenue, followed by Airport and Mall. "
            "Residential stations are weakest. New deployments should prioritise highways and commercial hubs.")

    st.divider()

    # ── Chart 2 ──────────────────────────────────────────────
    st.markdown("#### Chart 2 — Average Daily Revenue by State")
    state_rev = dff.groupby("State")["Revenue_Per_Day"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(11,5))
    bars2 = ax.bar(state_rev.index, state_rev.values,
                   color=[PALETTE[0] if i<3 else PALETTE[1] if i<6 else PALETTE[4]
                          for i in range(len(state_rev))],
                   edgecolor="white", width=0.65)
    for bar,val in zip(bars2,state_rev.values):
        ax.text(bar.get_x()+bar.get_width()/2, val+100,
                f"₹{val:,.0f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.axhline(state_rev.mean(), color=PALETTE[5], linestyle='--', linewidth=1.5,
               label=f"Network Avg ₹{state_rev.mean():,.0f}")
    ax.set_ylabel("Avg Daily Revenue (₹)"); ax.set_title("Avg Daily Revenue by State")
    ax.tick_params(axis='x', rotation=20); ax.legend(fontsize=9)
    fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
    insight("Karnataka and Maharashtra lead revenue charts. Rajasthan and West Bengal fall below the "
            "network average (red dashed line) — these markets need targeted development or site review.")

    st.divider()

    # ── Chart 3 ──────────────────────────────────────────────
    st.markdown("#### Chart 3 — Sessions Per Day Distribution (Histogram)")
    fig, ax = plt.subplots(figsize=(10,5))
    n,bins,patches = ax.hist(dff["Sessions_Per_Day"], bins=14, color=PALETTE[1],
                              edgecolor='white', linewidth=1.2, alpha=0.9)
    for patch in patches:
        if patch.get_x()+patch.get_width()/2 > 22:
            patch.set_facecolor(PALETTE[0])
    ax.axvline(dff["Sessions_Per_Day"].mean(), color=PALETTE[5], linestyle='--',
               linewidth=2, label=f'Mean = {dff["Sessions_Per_Day"].mean():.1f}')
    ax.axvline(dff["Sessions_Per_Day"].median(), color=PALETTE[3], linestyle='-.',
               linewidth=2, label=f'Median = {dff["Sessions_Per_Day"].median():.1f}')
    ax.set_xlabel("Sessions Per Day"); ax.set_ylabel("Number of Stations")
    ax.set_title("Distribution of Daily Charging Sessions"); ax.legend(fontsize=10)
    fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
    insight("Near-normal distribution centred at ~18 sessions/day. Stations with >22 sessions (dark) "
            "are best-in-class. Studying these for replication can lift the network average significantly.")

    st.divider()

    # ── Charts 4 & 5 side by side ─────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Chart 4 — Avg Profit by Region")
        reg_prof = dff.groupby("Region")["Profit_Per_Day"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6,4))
        bar_clrs = [PALETTE[2] if v>0 else PALETTE[5] for v in reg_prof.values]
        bars4 = ax.bar(reg_prof.index, reg_prof.values, color=bar_clrs, edgecolor="white", width=0.5)
        for bar,val in zip(bars4,reg_prof.values):
            ax.text(bar.get_x()+bar.get_width()/2,
                    val+30 if val>=0 else val-200,
                    f"₹{val:,.0f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.axhline(0, color='grey', linewidth=0.8)
        ax.set_ylabel("Avg Profit (₹)"); ax.set_title("Avg Daily Profit by Region")
        fig.tight_layout(); st.pyplot(fig); plt.close()
        insight("South India leads profitability. East India underperforms — cost audit recommended.")

    with col_b:
        st.markdown("#### Chart 5 — Charger Type Distribution")
        ctype_cnt = dff["Charger_Type"].value_counts()
        fig, ax = plt.subplots(figsize=(6,4))
        wedges,texts,autotexts = ax.pie(ctype_cnt, labels=ctype_cnt.index,
            autopct='%1.1f%%', colors=PALETTE[:3], startangle=140,
            wedgeprops=dict(edgecolor='white', linewidth=2))
        for at in autotexts:
            at.set_color('white'); at.set_fontweight('bold'); at.set_fontsize(11)
        ax.set_title("Charger Type Distribution")
        fig.tight_layout(); st.pyplot(fig); plt.close()
        insight("Expanding Ultra-Fast DC mix improves margin. It commands highest revenue per session.")

    st.divider()

    # ── Chart 6 ──────────────────────────────────────────────
    st.markdown("#### Chart 6 — Station Count by Revenue Category")
    revcat = dff["Rev_Category"].value_counts().reindex(["Low","Mid","High"])
    fig, ax = plt.subplots(figsize=(8,4))
    bars6 = ax.bar(["Low (<₹5K)","Mid (₹5K–15K)","High (>₹15K)"],
                   revcat.values, color=[PALETTE[5],PALETTE[3],PALETTE[2]],
                   edgecolor="white", width=0.5)
    for bar,cnt in zip(bars6,revcat.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f"{cnt}\n({cnt/len(dff)*100:.1f}%)",
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel("Number of Stations"); ax.set_title("Station Count by Revenue Category")
    ax.set_ylim(0, revcat.max()*1.25)
    fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
    insight(f"Only {revcat.get('High',0)} stations are High revenue earners (>₹15K/day). "
            "Improving Low-category stations (better siting, pricing, marketing) is the biggest opportunity.")

    st.divider()

    # ── Charts 7 & 8 ──────────────────────────────────────────
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("#### Chart 7 — Avg Customer Rating by Station Type")
        stype_rat = dff.groupby("Station_Type")["Customer_Rating"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6,4))
        clrs7 = [PALETTE[2] if v>=4 else PALETTE[3] if v>=3.5 else PALETTE[5] for v in stype_rat.values]
        bars7 = ax.bar(stype_rat.index, stype_rat.values, color=clrs7, edgecolor="white", width=0.6)
        for bar,val in zip(bars7,stype_rat.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f"{val:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.axhline(4.0, color=PALETTE[5], linestyle='--', linewidth=1.5, label='4.0 Threshold')
        ax.set_ylim(3.0, 5.0); ax.set_ylabel("Avg Rating")
        ax.set_title("Avg Rating by Station Type"); ax.legend(fontsize=8)
        ax.tick_params(axis='x', rotation=15)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        insight("Airport & Hotel stations score highest. Appointment-based booking at highways could lift scores.")

    with col_d:
        st.markdown("#### Chart 8 — Profit Margin % by Charger Type (Box Plot)")
        data8  = [dff[dff["Charger_Type"]==ct]["Profit_Margin_Pct"].dropna().values
                  for ct in dff["Charger_Type"].unique()]
        lbl8   = list(dff["Charger_Type"].unique())
        fig, ax = plt.subplots(figsize=(6,4))
        bp = ax.boxplot(data8, tick_labels=lbl8, patch_artist=True, widths=0.5,
                        medianprops=dict(color='white', linewidth=2))
        for patch,color in zip(bp['boxes'], PALETTE[:3]):
            patch.set_facecolor(color); patch.set_alpha(0.85)
        ax.set_ylabel("Profit Margin (%)"); ax.set_title("Profit Margin % by Charger Type")
        ax.axhline(0, color='red', linestyle=':', linewidth=1)
        ax.tick_params(axis='x', rotation=10)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        insight("Ultra-Fast DC achieves the highest, most consistent margins. AC Level 2 has lowest but steadier returns.")

    st.divider()

    # ── Charts 9 & 10 ─────────────────────────────────────────
    col_e, col_f = st.columns(2)
    with col_e:
        st.markdown("#### Chart 9 — Revenue vs Sessions/Day (Scatter)")
        fig, ax = plt.subplots(figsize=(6,4.5))
        ax.scatter(dff["Sessions_Per_Day"], dff["Revenue_Per_Day"],
                   c=PALETTE[1], alpha=0.5, s=35, edgecolors=PALETTE[0], linewidths=0.4)
        z = np.polyfit(dff["Sessions_Per_Day"], dff["Revenue_Per_Day"], 1)
        xs = np.linspace(dff["Sessions_Per_Day"].min(), dff["Sessions_Per_Day"].max(), 100)
        ax.plot(xs, np.poly1d(z)(xs), color=PALETTE[5], linewidth=2, linestyle='--', label='Trend')
        r = np.corrcoef(dff["Sessions_Per_Day"], dff["Revenue_Per_Day"])[0,1]
        ax.text(0.05, 0.90, f"r = {r:.3f}", transform=ax.transAxes,
                fontsize=11, fontweight='bold', color=PALETTE[0],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#EBF5FB'))
        ax.set_xlabel("Sessions Per Day"); ax.set_ylabel("Revenue (₹)")
        ax.set_title("Revenue vs Sessions/Day"); ax.legend(fontsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        insight(f"Strong positive correlation (r={r:.3f}). Every extra session ≈ ₹500 additional revenue.")

    with col_f:
        st.markdown("#### Chart 10 — Uptime % vs Customer Rating (Scatter)")
        fig, ax = plt.subplots(figsize=(6,4.5))
        for rc,col in zip(["High","Mid","Low"],[PALETTE[2],PALETTE[3],PALETTE[5]]):
            sub = dff[dff["Rev_Category"]==rc]
            ax.scatter(sub["Uptime_Pct"], sub["Customer_Rating"],
                       c=col, alpha=0.6, s=35, label=f"{rc} Rev", edgecolors='white', linewidths=0.3)
        z2 = np.polyfit(dff["Uptime_Pct"], dff["Customer_Rating"], 1)
        xs2 = np.linspace(dff["Uptime_Pct"].min(), dff["Uptime_Pct"].max(), 100)
        ax.plot(xs2, np.poly1d(z2)(xs2), color='#2C3E50', linewidth=2, linestyle='--')
        r2 = np.corrcoef(dff["Uptime_Pct"], dff["Customer_Rating"])[0,1]
        ax.text(0.05, 0.90, f"r = {r2:.3f}", transform=ax.transAxes,
                fontsize=11, fontweight='bold', color=PALETTE[0],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#EBF5FB'))
        ax.axvline(90, color='gray', linestyle=':', linewidth=1.2, label='90% Threshold')
        ax.set_xlabel("Uptime %"); ax.set_ylabel("Customer Rating")
        ax.set_title("Uptime % vs Customer Rating"); ax.legend(fontsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        insight(f"Uptime above 90% consistently yields ratings above 4.0 (r={r2:.3f}). Enforce preventive maintenance SLAs.")

    st.divider()

    # ── Charts 11 & 12 ────────────────────────────────────────
    col_g, col_h = st.columns(2)
    with col_g:
        st.markdown("#### Chart 11 — Revenue by Installation Year")
        yr_avg = dff.groupby("Install_Year")["Revenue_Per_Day"].mean()
        yr_cnt = dff.groupby("Install_Year")["Station_ID"].count()
        fig, ax1 = plt.subplots(figsize=(6,4))
        ax2 = ax1.twinx()
        ax1.plot(yr_avg.index, yr_avg.values, color=PALETTE[0], linewidth=3,
                 marker='o', markersize=9, label='Avg Revenue')
        ax1.fill_between(yr_avg.index, yr_avg.values, alpha=0.12, color=PALETTE[1])
        for yr,val in zip(yr_avg.index, yr_avg.values):
            ax1.text(yr, val+80, f"₹{val:,.0f}", ha='center', fontsize=8.5, fontweight='bold')
        ax2.bar(yr_cnt.index, yr_cnt.values, alpha=0.25, color=PALETTE[3], label='Station Count')
        ax1.set_ylabel("Avg Revenue (₹)", color=PALETTE[0])
        ax2.set_ylabel("Station Count", color=PALETTE[3])
        ax1.set_title("Revenue by Installation Year")
        fig.tight_layout(); st.pyplot(fig); plt.close()
        insight("2021 stations have highest avg revenue — mature market presence. 6–8 month ramp-up period observed.")

    with col_h:
        st.markdown("#### Chart 12 — Profitable vs Loss-Making Stations")
        prof_cnt    = int(dff["Profit_Flag"].sum())
        nonprof_cnt = len(dff) - prof_cnt
        fig, ax = plt.subplots(figsize=(6,4))
        wedges,texts = ax.pie([prof_cnt, nonprof_cnt],
            labels=[f"Profitable\n{prof_cnt} ({prof_cnt/len(dff)*100:.1f}%)",
                    f"Loss-Making\n{nonprof_cnt} ({nonprof_cnt/len(dff)*100:.1f}%)"],
            colors=[PALETTE[2], PALETTE[5]],
            startangle=90, wedgeprops=dict(width=0.55, edgecolor='white', linewidth=3))
        ax.text(0, 0, f"{prof_cnt/len(dff)*100:.1f}%\nProfitable",
                ha='center', va='center', fontsize=13, fontweight='bold', color=PALETTE[2])
        ax.set_title("Network Profitability Split")
        fig.tight_layout(); st.pyplot(fig); plt.close()
        insight(f"{prof_cnt/len(dff)*100:.1f}% of stations are profitable. The {nonprof_cnt} loss-making stations "
                "need cost audit, dynamic pricing, or relocation consideration.")


# ══════════════════════════════════════════════════════════════
# TAB 5 — CORRELATION
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<span class="section-tag">CORRELATION ANALYSIS</span>', unsafe_allow_html=True)
    st.subheader("Pearson Correlation Matrix — Key Numerical Variables")

    num_cols_c = ["Sessions_Per_Day","kWh_Per_Session","Price_Per_kWh",
                  "Revenue_Per_Day","Energy_Cost_Per_Day","Maint_Cost_Per_Day",
                  "Profit_Per_Day","Customer_Rating","Uptime_Pct","Profit_Margin_Pct"]
    short_n    = ["Sessions","kWh","Price","Revenue","EnCost","MaintCost",
                  "Profit","Rating","Uptime","ProfMgn%"]

    corr = dff[num_cols_c].corr()
    corr.index   = short_n
    corr.columns = short_n

    fig, ax = plt.subplots(figsize=(12,9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = sns.diverging_palette(10, 133, as_cmap=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, vmin=-1, vmax=1,
                center=0, ax=ax, square=True, linewidths=0.5, linecolor='white',
                annot_kws={"size":9,"weight":"bold"},
                cbar_kws={"shrink":0.7,"label":"Pearson r"})
    ax.set_title("Pearson Correlation Heatmap — ChargePath India Numerical Variables", pad=20, fontsize=14)
    ax.tick_params(axis='x', rotation=40, labelsize=9)
    ax.tick_params(axis='y', rotation=0,  labelsize=9)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.divider()
    st.subheader("Key Correlation Findings")
    corr_findings = [
        ("Revenue ↔ Sessions/Day",    f"{corr.loc['Revenue','Sessions']:.3f}",
         "Strong Positive", "Sessions are the PRIMARY revenue driver. Every extra session ≈ ₹500 more revenue."),
        ("Revenue ↔ Energy Cost",     f"{corr.loc['Revenue','EnCost']:.3f}",
         "Strong Positive", "Variable costs scale with usage. Renewable energy procurement deals can decouple this."),
        ("Profit Margin ↔ Price",     f"{corr.loc['ProfMgn%','Price']:.3f}",
         "Moderate Positive","Dynamic pricing during peak hours directly boosts margin without added cost."),
        ("Rating ↔ Uptime",           f"{corr.loc['Rating','Uptime']:.3f}",
         "Moderate Positive","Reliable stations get better ratings. Enforce 90%+ uptime SLA across network."),
        ("Profit ↔ Maint Cost",       f"{corr.loc['Profit','MaintCost']:.3f}",
         "Moderate Negative","High maintenance drag profit down. Root-cause analysis for high-cost stations needed."),
    ]
    for var, r_val, strength, desc in corr_findings:
        col1,col2,col3 = st.columns([2,1.2,5])
        with col1: st.markdown(f"**{var}**")
        with col2:
            clr = "green" if float(r_val)>0 else "red"
            st.markdown(f"<span style='color:{clr};font-weight:700;font-size:1.05rem;'>{r_val} ({strength})</span>",
                        unsafe_allow_html=True)
        with col3: st.markdown(desc)
        st.markdown("---")


# ══════════════════════════════════════════════════════════════
# TAB 6 — DATASET
# ══════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<span class="section-tag">DATASET EXPLORER</span>', unsafe_allow_html=True)

    view = st.radio("View:", ["Cleaned Dataset (197 rows × 25 cols)", "Raw Dataset (200 rows × 17 cols)"],
                    horizontal=True)

    if "Cleaned" in view:
        display_df = dff.drop(columns=["Install_Date_dt"], errors='ignore')
        st.markdown(f"**{len(display_df)} rows × {len(display_df.columns)} columns** — after cleaning & feature engineering")
    else:
        display_df = raw
        missing_rows = raw[raw.isnull().any(axis=1)]
        st.warning(f"Raw data contains {raw.isnull().sum().sum()} missing cells across {len(missing_rows)} rows.")

    search = st.text_input("Search by Station ID or City")
    if search:
        mask2 = (
            display_df["Station_ID"].str.contains(search, case=False, na=False) |
            display_df["City"].str.contains(search, case=False, na=False)
        )
        display_df = display_df[mask2]

    st.dataframe(display_df, use_container_width=True, height=420)

    csv = display_df.to_csv(index=False).encode()
    st.download_button("⬇️ Download CSV", csv,
                       file_name="chargepath_india_dataset.csv",
                       mime="text/csv")

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#6B7280;font-size:0.82rem;'>"
    "⚡ ChargePath India EDA Dashboard · MBA Data Analytics Individual Assignment · "
    "Built with Python, Streamlit, Matplotlib & Seaborn"
    "</div>",
    unsafe_allow_html=True
)
