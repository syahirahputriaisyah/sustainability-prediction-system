from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime
from models_logic import FinanceModels

app = Flask(__name__)
CORS(app)

# Cache data & models
@app.before_request
def refresh_cache():
    # In production, this data should be loaded once
    pass

def load_data():
    df = pd.read_csv('student_financial_data.csv')
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    df = load_data()
    # Totals
    total_spent = int(df[df['Tipe'] == 'Pengeluaran']['Jumlah'].sum())
    total_income = int(df[df['Tipe'] == 'Pemasukan']['Jumlah'].sum())
    
    avg_monthly_exp = int(df[df['Tipe'] == 'Pengeluaran'].groupby(df['Tanggal'].dt.month)['Jumlah'].sum().mean())
    tx_count = len(df)
    
    # Timeline
    monthly_exp = df[df['Tipe'] == 'Pengeluaran'].groupby(pd.Grouper(key='Tanggal', freq='M'))['Jumlah'].sum().reset_index()
    monthly_inc = df[df['Tipe'] == 'Pemasukan'].groupby(pd.Grouper(key='Tanggal', freq='M'))['Jumlah'].sum().reset_index()
    
    # Merge for Cashflow
    cashflow = pd.merge(monthly_exp, monthly_inc, on='Tanggal', how='outer', suffixes=('_exp', '_inc')).fillna(0)
    cashflow = cashflow.sort_values('Tanggal')
    
    timeline_data = {
        'dates': cashflow['Tanggal'].dt.strftime('%b %Y').tolist(),
        'exp': cashflow['Jumlah_exp'].tolist(),
        'inc': cashflow['Jumlah_inc'].tolist(),
        'net': (cashflow['Jumlah_inc'] - cashflow['Jumlah_exp']).tolist()
    }

    # Category breakdowns (Pengeluaran)
    exp_cat = df[df['Tipe'] == 'Pengeluaran'].groupby('Kategori')['Jumlah'].sum().reset_index()
    inc_cat = df[df['Tipe'] == 'Pemasukan'].groupby('Kategori')['Jumlah'].sum().reset_index()

    return jsonify({
        'summary': {
            'total_spent': total_spent,
            'total_income': total_income,
            'avg_monthly_exp': avg_monthly_exp,
            'tx_count': tx_count,
            'net_balance': total_income - total_spent
        },
        'trends': timeline_data,
        'categories': {
            'exp': {'labels': exp_cat['Kategori'].tolist(), 'values': exp_cat['Jumlah'].tolist()},
            'inc': {'labels': inc_cat['Kategori'].tolist(), 'values': inc_cat['Jumlah'].tolist()}
        }
    })

@app.route('/api/ai_insights')
def get_ai_insights():
    df = load_data()
    models = FinanceModels()
    
    # Regression for Expense
    Xe, ye, base_e = models.prepare_regression_data(df, tipe='Pengeluaran')
    reg_e = models.train_regression(Xe, ye)
    pred_e = int(models.predict_next(reg_e, base_e['MonthIndex'].max(), base_e['Jumlah'].iloc[-1]))

    # Regression for Income
    Xi, yi, base_i = models.prepare_regression_data(df, tipe='Pemasukan')
    reg_i = models.train_regression(Xi, yi)
    pred_i = int(models.predict_next(reg_i, base_i['MonthIndex'].max(), base_i['Jumlah'].iloc[-1]))
    
    # Clustering
    persona, profile = models.perform_clustering(df)
    
    return jsonify({
        'pred_exp': pred_e,
        'pred_inc': pred_i,
        'persona': persona,
        'profile': profile
    })

@app.route('/api/simulate', methods=['POST'])
def simulate():
    input_data = request.json
    income = float(input_data.get('income'))
    categories = input_data.get('categories')
    
    total_plan = sum(categories.values())
    savings = income - total_plan
    
    # Create temp dataframe for simulation persona
    data = []
    # Add simulated income entry
    data.append({'Kategori': 'Simulasi Income', 'Jumlah': income, 'Tanggal': datetime.now(), 'Tipe': 'Pemasukan'})
    # Add simulated expense entries
    for k, v in categories.items():
        data.append({'Kategori': k, 'Jumlah': v, 'Tanggal': datetime.now(), 'Tipe': 'Pengeluaran'})
    
    df_sim = pd.DataFrame(data)
    models = FinanceModels()
    persona, _ = models.perform_clustering(df_sim)
    
    return jsonify({
        'total_plan': total_plan,
        'savings': savings,
        'persona': persona,
        'status': 'Defisit' if savings < 0 else 'Sehat'
    })

@app.route('/api/add_transaction', methods=['POST'])
def add_transaction():
    input_data = request.json
    # Format for CSV: Tanggal, Kategori, Jumlah, Tipe
    new_entry = {
        'Tanggal': datetime.now().strftime('%Y-%m-%d'),
        'Kategori': input_data.get('kategori'),
        'Jumlah': float(input_data.get('jumlah')),
        'Tipe': input_data.get('tipe')
    }
    
    df = pd.read_csv('student_financial_data.csv')
    # Use pd.concat for higher reliability in new pandas versions
    new_row = pd.DataFrame([new_entry])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv('student_financial_data.csv', index=False)
    
    return jsonify({'status': 'Success', 'message': 'Transaksi berhasil dicatat!'})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
