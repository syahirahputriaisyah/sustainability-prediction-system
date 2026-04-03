import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class FinanceModels:
    def __init__(self):
        self.reg_model = LinearRegression()
        self.cluster_model = KMeans(n_clusters=3, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        
    def prepare_regression_data(self, df, tipe='Pengeluaran'):
        # Filter by type
        filtered = df[df['Tipe'] == tipe].copy()
        filtered['Tanggal'] = pd.to_datetime(filtered['Tanggal'])
        
        monthly = filtered.groupby(pd.Grouper(key='Tanggal', freq='M')).agg({
            'Jumlah': 'sum'
        }).reset_index()
        
        monthly['MonthIndex'] = range(len(monthly))
        monthly['Prev_Amount'] = monthly['Jumlah'].shift(1)
        monthly = monthly.dropna()
        
        X = monthly[['MonthIndex', 'Prev_Amount']]
        y = monthly['Jumlah']
        
        return X, y, monthly

    def train_regression(self, X, y):
        model = LinearRegression()
        model.fit(X, y)
        return model

    def predict_next(self, model, last_month_index, last_amount):
        X_next = np.array([[last_month_index + 1, last_amount]])
        return model.predict(X_next)[0]

    def perform_clustering(self, df):
        # Separate Spend and Income
        total_spent = df[df['Tipe'] == 'Pengeluaran'].groupby('Kategori')['Jumlah'].sum()
        total_income = df[df['Tipe'] == 'Pemasukan']['Jumlah'].sum()
        total_exp = total_spent.sum()
        
        # Calculate percentages for spending behavior
        exp_categories = ['Makan', 'Transportasi', 'Pendidikan', 'Hiburan', 'Lainnya']
        spend_profile = {}
        for cat in exp_categories:
            spend_profile[cat] = (total_spent.get(cat, 0) / (total_exp if total_exp > 0 else 1)) * 100
        
        # Savings ratio
        savings_ratio = ((total_income - total_exp) / (total_income if total_income > 0 else 1)) * 100
        spend_profile['RasioTabungan'] = savings_ratio
        
        # Food and Fun percentages for clustering
        food_perc = spend_profile.get('Makan', 0)
        fun_perc = spend_profile.get('Hiburan', 0)
        
        # Personas Centroids: [Food %, Fun %, Savings %]
        personas = {
            "Seimbang": [30, 20, 20],
            "Impulsif (Boros)": [10, 60, 5],
            "Hemat (Frugal)": [40, 5, 60]
        }
        
        user_vector = [food_perc, fun_perc, savings_ratio]
        
        # Calculate Euclidean Distance to each persona
        def calculate_dist(v1, v2):
            return np.sqrt(sum((a - b)**2 for a, b in zip(v1, v2)))
        
        distances = {name: calculate_dist(user_vector, target) for name, target in personas.items()}
        
        # Pick the closest persona
        closest_persona = min(distances, key=distances.get)
        
        return closest_persona, spend_profile
