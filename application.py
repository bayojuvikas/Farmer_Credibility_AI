# =========================================
# Credit Bridge Streamlit Dashboard
# =========================================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

st.set_page_config(page_title="Credit Bridge Dashboard", layout="wide")

st.title("💳 Credit Bridge: Risk & Opportunity Dashboard")

# -----------------------------
# Step 1: Upload CSVs or use synthetic data
# -----------------------------
st.sidebar.header("Upload Input Data (CSV)")
uploaded_financial = st.sidebar.file_uploader("Financial Data", type=["csv"])
uploaded_agri = st.sidebar.file_uploader("Agricultural Data", type=["csv"])
uploaded_alt = st.sidebar.file_uploader("Alternative Data", type=["csv"])
uploaded_consent = st.sidebar.file_uploader("Consent/Compliance Data", type=["csv"])
uploaded_land = st.sidebar.file_uploader("Land Ownership & Tenure Data", type=["csv"])

num_samples = 1000

def generate_synthetic_data():
    np.random.seed(42)
    financial_data = pd.DataFrame({
        'bank_balance': np.random.randint(500, 50000, num_samples),
        'loan_amount': np.random.randint(1000, 20000, num_samples),
        'previous_defaults': np.random.randint(0,5, num_samples),
        'credit_score': np.random.randint(300,850, num_samples),
        'target_default': np.random.randint(0,2, num_samples)
    })
    agri_data = pd.DataFrame({
        'crop_yield': np.random.randint(100, 1000, num_samples),
        'soil_ph': np.random.uniform(5.0, 8.0, num_samples),
        'rainfall_mm': np.random.randint(50, 400, num_samples),
        'temperature_c': np.random.uniform(15, 40, num_samples),
        'target_yield_category': np.random.randint(0,3, num_samples)
    })
    alt_data = pd.DataFrame({
        'utility_payments': np.random.randint(50, 1000, num_samples),
        'mobile_usage': np.random.randint(100, 5000, num_samples),
        'digital_txns': np.random.randint(0,50, num_samples),
        'target_risk': np.random.randint(0,2, num_samples)
    })
    consent_data = pd.DataFrame({
        'consent_given': np.random.randint(0,2, num_samples),
        'audit_events': np.random.randint(0,20, num_samples),
        'data_access_count': np.random.randint(0,10, num_samples),
        'target_compliance_risk': np.random.randint(0,2, num_samples)
    })
    land_data = pd.DataFrame({
        'land_size_acres': np.random.uniform(0.5, 10.0, num_samples),
        'land_value_inr': np.random.randint(50000, 1000000, num_samples),
        'land_type': np.random.randint(0,2, num_samples),
        'title_verified': np.random.randint(0,2, num_samples),
        'zoning_category': np.random.randint(0,2, num_samples),
        'target_land_risk': np.random.randint(0,2, num_samples)
    })
    return financial_data, agri_data, alt_data, consent_data, land_data

financial_data = pd.read_csv(uploaded_financial) if uploaded_financial else generate_synthetic_data()[0]
agri_data = pd.read_csv(uploaded_agri) if uploaded_agri else generate_synthetic_data()[1]
alt_data = pd.read_csv(uploaded_alt) if uploaded_alt else generate_synthetic_data()[2]
consent_data = pd.read_csv(uploaded_consent) if uploaded_consent else generate_synthetic_data()[3]
land_data = pd.read_csv(uploaded_land) if uploaded_land else generate_synthetic_data()[4]

st.success("✅ Input data loaded successfully!")

# -----------------------------
# Step 2: Train Models
# -----------------------------
st.subheader("Training Models... This may take a few seconds")

# Financial Model (XGBoost)
X_fin = financial_data.drop('target_default', axis=1)
y_fin = financial_data['target_default']
X_train_fin, X_test_fin, y_train_fin, y_test_fin = train_test_split(X_fin, y_fin, test_size=0.2, random_state=42)
xgb_fin = XGBClassifier(eval_metric='auc', use_label_encoder=False)
xgb_fin.fit(X_train_fin, y_train_fin)
financial_score = 1 - xgb_fin.predict_proba(X_test_fin)[:,1]  # 0-1 (higher = better)

# Agricultural Model (Random Forest)
X_agri = agri_data.drop('target_yield_category', axis=1)
y_agri = agri_data['target_yield_category']
X_train_agri, X_test_agri, y_train_agri, y_test_agri = train_test_split(X_agri, y_agri, test_size=0.2, random_state=42)
rf_agri = RandomForestClassifier(n_estimators=200, random_state=42)
rf_agri.fit(X_train_agri, y_train_agri)
yield_mapping = {0:0.33, 1:0.66, 2:1.0}
agri_score = np.array([yield_mapping[i] for i in rf_agri.predict(X_test_agri)])  # 0-1

# Alternative Model (XGBoost)
X_alt = alt_data.drop('target_risk', axis=1)
y_alt = alt_data['target_risk']
X_train_alt, X_test_alt, y_train_alt, y_test_alt = train_test_split(X_alt, y_alt, test_size=0.2, random_state=42)
xgb_alt = XGBClassifier(eval_metric='auc', use_label_encoder=False)
xgb_alt.fit(X_train_alt, y_train_alt)
alt_score = 1 - xgb_alt.predict_proba(X_test_alt)[:,1]  # 0-1

# Consent/Compliance Model (Neural Network)
X_consent = consent_data.drop('target_compliance_risk', axis=1)
y_consent = consent_data['target_compliance_risk']
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_consent, y_consent, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_c_scaled = scaler.fit_transform(X_train_c)
X_test_c_scaled = scaler.transform(X_test_c)
nn_model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_c_scaled.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train_c_scaled, y_train_c, epochs=20, batch_size=32, validation_split=0.2, verbose=0)
consent_score = 1 - nn_model.predict(X_test_c_scaled).flatten()  # 0-1

# Land Risk Model (Gradient Boosting)
X_land = land_data.drop('target_land_risk', axis=1)
y_land = land_data['target_land_risk']
X_train_land, X_test_land, y_train_land, y_test_land = train_test_split(X_land, y_land, test_size=0.2, random_state=42)
gb_land = GradientBoostingClassifier(random_state=42)
gb_land.fit(X_train_land, y_train_land)
land_proba = gb_land.predict_proba(X_test_land)[:,1]
land_score = 1 - land_proba  # 0-1 (higher = lower risk)

# Align lengths (truncate to smallest test-slice)
min_len = min(len(financial_score), len(agri_score), len(alt_score), len(consent_score), len(land_score))
financial_score = financial_score[:min_len]
agri_score = agri_score[:min_len]
alt_score = alt_score[:min_len]
consent_score = consent_score[:min_len]
land_score = land_score[:min_len]

# -----------------------------
# Step 3: Unified Score (weights sum to 1.0)
# -----------------------------
weights = {
    'financial': 0.35,
    'agri': 0.25,
    'alternative': 0.15,
    'land': 0.15,
    'consent': 0.10
}

unified_score = (
    financial_score * weights['financial'] +
    agri_score * weights['agri'] +
    alt_score * weights['alternative'] +
    land_score * weights['land'] +
    consent_score * weights['consent']
)  # still 0-1

# Build internal dashboard (keep decimal cols for computation now)
dashboard = pd.DataFrame({
    'Financial_Score': financial_score,
    'Agricultural_Score': agri_score,
    'Alternative_Score': alt_score,
    'Consent_Score': consent_score,
    'Land_Score': land_score,
    'Unified_Risk_Opportunity_Score': unified_score
})

# -----------------------------
# Step 4: Convert to 0-100 units and add labels (integers)
# -----------------------------
# Convert to integer 0-100 units (no decimals)
dashboard['Financial_Score_100'] = (dashboard['Financial_Score'] * 100).round(0).astype(int)
dashboard['Agricultural_Score_100'] = (dashboard['Agricultural_Score'] * 100).round(0).astype(int)
dashboard['Alternative_Score_100'] = (dashboard['Alternative_Score'] * 100).round(0).astype(int)
dashboard['Consent_Score_100'] = (dashboard['Consent_Score'] * 100).round(0).astype(int)
dashboard['Land_Score_100'] = (dashboard['Land_Score'] * 100).round(0).astype(int)
dashboard['Unified_Score_100'] = (dashboard['Unified_Risk_Opportunity_Score'] * 100).round(0).astype(int)

# Rank using the 0-100 unified score
dashboard['Rank'] = dashboard['Unified_Score_100'].rank(ascending=False, method='min').astype(int)

# Categorize
def categorize_score(score):
    if score >= 85:
        return "Excellent (Low Risk)"
    elif score >= 70:
        return "Good (Moderate Risk)"
    elif score >= 55:
        return "Fair (Borderline)"
    else:
        return "Poor (High Risk)"

dashboard['Credibility_Rating'] = dashboard['Unified_Score_100'].apply(categorize_score)

st.success("✅ Models trained and unified scores calculated!")

# -----------------------------
# Step 5: Prepare display/export dataframe (remove decimal columns)
# -----------------------------
display_cols = [
    'Financial_Score_100', 'Agricultural_Score_100',
    'Alternative_Score_100', 'Consent_Score_100',
    'Land_Score_100', 'Unified_Score_100',
    'Credibility_Rating', 'Rank'
]

dashboard_display = dashboard[display_cols].copy()

# -----------------------------
# Step 6: Display Dashboard (styled)
# -----------------------------
st.subheader("Top 10 Clients/Farmers by Unified Score (0–100 units)")

# Color mapping for rating
def color_risk(val):
    if "Excellent" in val:
        return 'background-color: #66BB6A; color: white; font-weight: bold;'
    elif "Good" in val:
        return 'background-color: #FFD54F; color: black; font-weight: bold;'
    elif "Fair" in val:
        return 'background-color: #FFA726; color: white; font-weight: bold;'
    else:
        return 'background-color: #EF5350; color: white; font-weight: bold;'

# Apply styling to the Credibility_Rating column only
# -----------------------------
# Step 6: Styling
# -----------------------------
# Sort first
top10_dashboard = dashboard_display.sort_values('Unified_Score_100', ascending=False).head(10)

# Apply styling
styled = top10_dashboard.style.applymap(lambda v: '', subset=[c for c in top10_dashboard.columns if c != 'Credibility_Rating'])
styled = styled.applymap(color_risk, subset=['Credibility_Rating'])

# Show top 10 table (styled)
st.dataframe(styled)

# Also show a simple table without styling for copy/export


# -----------------------------
# Step 7: Visualization (bar chart using 0-100 units)
# -----------------------------
st.subheader("Top 20 Clients/Farmers Visualization (0–100)")

top_dashboard = dashboard_display.sort_values('Unified_Score_100', ascending=False).head(20).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(12,6))
sns.barplot(x=top_dashboard.index + 1, y=top_dashboard['Unified_Score_100'], palette="viridis", ax=ax)
ax.set_xlabel("Rank (1 = Best)")
ax.set_ylabel("Unified Score (0 - 100)")
ax.set_title("Top 20 Clients/Farmers")
ax.set_ylim(0, 100)
plt.xticks(rotation=0)
plt.tight_layout()
st.pyplot(fig)

# -----------------------------
# Step 4.1: Compact Badges and Individual Gauge
# -----------------------------

st.subheader("🎯 Farmer Credibility Insights")

# Select a farmer by index
selected_farmer = st.selectbox(
    "Select Farmer/Client ID to View Details",
    options=dashboard.index[:50],  # show first 50 for simplicity
    index=0
)

farmer_row = dashboard.loc[selected_farmer]

# Badge Style Function
def badge_html(text):
    color_map = {
        "Excellent": "#4CAF50",
        "Good": "#FFC107",
        "Fair": "#FF9800",
        "Poor": "#F44336"
    }
    color = next((v for k, v in color_map.items() if k in text), "#BDBDBD")
    return f"<span style='background-color:{color}; color:white; padding:6px 12px; border-radius:15px; font-weight:bold;'>{text}</span>"

# Display key info
st.markdown(f"""
### 🧾 Farmer ID: {selected_farmer}
**Unified Credibility Score:** {farmer_row['Unified_Score_100']}  
**Category:** {badge_html(farmer_row['Credibility_Rating'])}
""", unsafe_allow_html=True)

# Gauge Chart
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=farmer_row['Unified_Score_100'],
    title={'text': "Credibility Score (0–100)"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': '#2196F3'},
        'steps': [
            {'range': [0, 55], 'color': '#EF5350'},
            {'range': [55, 70], 'color': '#FFA726'},
            {'range': [70, 85], 'color': '#FFD54F'},
            {'range': [85, 100], 'color': '#66BB6A'}
        ],
    }
))
st.plotly_chart(fig_gauge, use_container_width=True)

# -----------------------------
# Step 8: Download Reports (Excel)
# -----------------------------
st.subheader("Download Reports")
excel_filename = "CreditBridge_Dashboard.xlsx"

# Export only the display dataframe (no decimal columns)
dashboard_display.to_excel(excel_filename, index=False)

with open(excel_filename, "rb") as f:
    st.download_button("⬇️ Download Excel Report", data=f, file_name=excel_filename)
    st.download_button("⬇️ Download Synthetic Data (ZIP)", data=pd.ExcelWriter("synthetic_data.xlsx").path if False else None, file_name="synthetic_data.zip")


