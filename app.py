import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from models_logic import FinanceModels

# --- Configuration & Styling ---
st.set_page_config(
    page_title="Fin-AI Admin Dashboard",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Admin Dashboard Look
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Scaling */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Card Container */
    .metric-card {
        background: #161b22;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #30363d;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #58a6ff;
    }
    
    /* Big Metric Styling */
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #58a6ff;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* AI Insight Badge */
    .insight-badge {
        background: linear-gradient(135deg, #1f6feb 0%, #3e1fbe 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #58a6ff;
        margin: 1rem 0;
    }
    
    /* Custom divider */
    .hr-custom {
        margin: 2rem 0;
        border: 0;
        border-top: 1px solid #30363d;
    }
    
    /* Chart Container */
    .chart-container {
        background: #161b22;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)

# Helper function for card-like metrics
def styled_metric(label, value, delta=None):
    delta_html = f'<div style="color: #3fb950; font-size: 0.8rem;">▲ {delta}</div>' if delta else ""
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)

# --- App Logic ---
def main():
    # Sidebar Header
    st.sidebar.markdown("<h1 style='text-align: center; color: #58a6ff;'>⚡ Fin-AI</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center; color: #8b949e;'>Admin Dashboard v1.0</p>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    app_mode = st.sidebar.radio("MAIN MENU", ["📊 Dashboard Overview", "🔮 AI Predictions", "🕹️ User Simulation"])
    
    # Load default data
    @st.cache_data
    def load_data():
        try:
            return pd.read_csv('student_financial_data.csv')
        except:
            return None

    df_base = load_data()
    if df_base is None:
        st.error("Dataset not found. Please generate data first.")
        return
    df_base['Tanggal'] = pd.to_datetime(df_base['Tanggal'])

    if app_mode == "📊 Dashboard Overview":
        render_analysis(df_base)
    elif app_mode == "🔮 AI Predictions":
        render_predictions_page(df_base)
    else:
        render_simulation()

def render_analysis(df):
    st.markdown("## 📊 Financial Overview")
    st.markdown("Welcome back, Chief! Here's your financial status at a glance.")
    
    # Top Stats Row
    total_spent = df['Jumlah'].sum()
    avg_monthly = df.groupby(df['Tanggal'].dt.month)['Jumlah'].sum().mean()
    tx_count = len(df)
    income_avg = df['Pemasukan_Bulanan'].mean()

    m1, m2, m3, m4 = st.columns(4)
    with m1: styled_metric("TOTAL EXPENSE", f"Rp {total_spent:,.0f}")
    with m2: styled_metric("AVG MONTHLY", f"Rp {avg_monthly:,.0f}")
    with m3: styled_metric("TOTAL TRANSACTIONS", f"{tx_count}")
    with m4: styled_metric("AVG INCOME", f"Rp {income_avg:,.0f}")

    # Charts Row
    st.markdown("<div class='hr-custom'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown("#### 📈 Spending Trends")
        monthly_df = df.groupby(pd.Grouper(key='Tanggal', freq='M'))['Jumlah'].sum().reset_index()
        fig_time = px.line(monthly_df, x='Tanggal', y='Jumlah', markers=True)
        fig_time.update_traces(line_color='#58a6ff', fill='tozeroy', fillcolor='rgba(88, 166, 255, 0.1)')
        fig_time.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=20, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_time, use_container_width=True)
        
    with c2:
        st.markdown("#### 🥧 Category Distribution")
        fig_pie = px.pie(df, values='Jumlah', names='Kategori', hole=0.6)
        fig_pie.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=20, b=0), paper_bgcolor="rgba(0,0,0,0)", showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)

def render_predictions_page(df):
    st.markdown("## 🔮 AI Predictions & Persona Analysis")
    
    models = FinanceModels()
    X, y, monthly = models.prepare_regression_data(df)
    models.train_regression(X, y)
    
    last_idx = monthly['MonthIndex'].max()
    last_spent = monthly['Jumlah'].iloc[-1]
    prediction = models.predict_next_month(last_idx, last_spent)
    persona, profile = models.perform_clustering(df)

    # UI for Predictions
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        st.markdown(f"""
        <div class='insight-badge'>
            <h3>👤 User Persona</h3>
            <h2 style='color: #ffca28;'>{persona}</h2>
            <p>Based on your historical clusters, you follow a <b>{persona}</b> pattern of spending.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Breakdown Radar-like Table
        st.write("#### Category Breakdown (%)")
        df_profile = pd.DataFrame(list(profile.items()), columns=['Category', 'Value'])
        st.table(df_profile)

    with col_p2:
        st.markdown(f"""
        <div class='metric-card' style='border-left: 5px solid #3fb950;'>
            <div class='metric-label'>NEXT MONTH PREDICTION</div>
            <div class='metric-value'>Rp {prediction:,.0f}</div>
            <p style='color: #8b949e;'>Predicted based on your consumption velocity and lag metrics.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Prediction Chart (Historical + Future point)
        hist_y = monthly['Jumlah'].tolist()
        hist_x = monthly['Tanggal'].dt.strftime('%b %Y').tolist()
        
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=hist_x, y=hist_y, name='Historical', line=dict(color='#58a6ff', width=2)))
        fig_pred.add_trace(go.Scatter(x=[hist_x[-1], "Bulan Depan"], y=[hist_y[-1], prediction], 
                                     name='AI Prediction', line=dict(color='#ffca28', dash='dash')))
        fig_pred.update_layout(template="plotly_dark", height=300, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pred, use_container_width=True)

def render_simulation():
    st.subheader("🕹️ Simulasi Input User")
    st.write("Masukkan rencana pengeluaran Anda untuk mendapatkan analisis AI.")

    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            income = st.number_input("Pemasukan Bulanan (Rp)", value=2000000, step=100000)
            makan = st.number_input("Budget Makan (Rp)", value=1000000, step=50000)
            transport = st.number_input("Budget Transportasi (Rp)", value=200000, step=10000)
        with col2:
            hiburan = st.number_input("Budget Hiburan (Rp)", value=500000, step=50000)
            edu = st.number_input("Budget Pendidikan (Rp)", value=100000, step=10000)
            lain = st.number_input("Biaya Lainnya (Rp)", value=100000, step=10000)
        
        submitted = st.form_submit_button("Analisis AI")

    if submitted:
        # Create temp dataframe for simulation
        data = []
        categories = {'Makan': makan, 'Transportasi': transport, 'Hiburan': hiburan, 'Pendidikan': edu, 'Lainnya': lain}
        for cat, val in categories.items():
            data.append({'Kategori': cat, 'Jumlah': val, 'Tanggal': datetime.now(), 'Pemasukan_Bulanan': income})
        
        df_sim = pd.DataFrame(data)
        df_sim['Tanggal'] = pd.to_datetime(df_sim['Tanggal'])
        
        models = FinanceModels()
        persona, profile = models.perform_clustering(df_sim)
        
        total_plan = sum(categories.values())
        savings = income - total_plan
        
        st.markdown("<div class='hr-custom'></div>", unsafe_allow_html=True)
        cc1, cc2 = st.columns(2)
        with cc1:
            styled_metric("PLANNED EXPENSE", f"Rp {total_plan:,.0f}")
            
            if savings < 0:
                color = "#f85149"
                msg = "🚨 BUDGET DEFICIT!"
            else:
                color = "#3fb950"
                msg = "✅ BUDGET SAFE"
                
            st.markdown(f"""
            <div class='metric-card' style='border-top: 4px solid {color};'>
                <div class='metric-label'>PROJECTED SAVINGS</div>
                <div class='metric-value' style='color: {color};'>Rp {savings:,.0f}</div>
                <p>{msg}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class='insight-badge'>
                <h4>🤖 AI Simulation Insight:</h4>
                <p>Based on this simulation, you are categorized as <b>{persona}</b>.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with cc2:
            fig = px.bar(x=list(categories.keys()), y=list(categories.values()), 
                        labels={'x': 'Kategori', 'y': 'Jumlah (Rp)'},
                        title="Budget Allocation Plan")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
