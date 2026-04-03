import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_mock_data(n_months=12):
    np.random.seed(42)
    exp_categories = ['Makan', 'Transportasi', 'Pendidikan', 'Hiburan', 'Lainnya']
    inc_categories = ['Uang Saku', 'Gaji/Part-time', 'Beasiswa', 'Lainnya']
    
    data = []
    current_date = datetime(2025, 1, 1)
    
    for month in range(n_months):
        # Generate Multiple Income Transactions for the month
        monthly_incomes = [
            ('Uang Saku', 2500000 + np.random.normal(0, 50000)),
            ('Gaji/Part-time', np.random.choice([0, 500000, 1000000], p=[0.4, 0.4, 0.2])),
            ('Beasiswa', np.random.choice([0, 1500000], p=[0.9, 0.1]))
        ]
        
        for inc_cat, amount in monthly_incomes:
            if amount > 0:
                data.append({
                    'Tanggal': current_date + timedelta(days=1), # Usually start of month
                    'Kategori': inc_cat,
                    'Jumlah': round(amount, -3),
                    'Tipe': 'Pemasukan'
                })
        
        # Total income for this month to use for Persona logic
        total_monthly_income = sum([a for _, a in monthly_incomes])
        
        # Generate Expenses
        for cat in exp_categories:
            num_tx = np.random.randint(5, 15)
            for _ in range(num_tx):
                amount = 0
                if cat == 'Makan': amount = np.random.gamma(shape=5, scale=5000)
                elif cat == 'Transportasi': amount = np.random.uniform(5000, 30000)
                elif cat == 'Pendidikan': amount = np.random.choice([0, 500000], p=[0.9, 0.1])
                elif cat == 'Hiburan': amount = np.random.exponential(scale=80000)
                else: amount = np.random.uniform(10000, 50000)
                
                days_to_add = np.random.randint(1, 28)
                tx_date = current_date + timedelta(days=days_to_add)
                
                data.append({
                    'Tanggal': tx_date,
                    'Kategori': cat,
                    'Jumlah': round(amount, -3),
                    'Tipe': 'Pengeluaran'
                })
        
        current_date = (current_date + timedelta(days=32)).replace(day=1)
        
    df = pd.DataFrame(data)
    df = df.sort_values('Tanggal').reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = generate_mock_data()
    df.to_csv('student_financial_data.csv', index=False)
    print("Dataset generated successfully: student_financial_data.csv")
