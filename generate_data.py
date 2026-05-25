import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_ecommerce_data(num_customers=1000, num_records=5000):
    np.random.seed(42)
    
    # 1. Generate Customers
    customers = pd.DataFrame({
        'customer_id': range(1, num_customers + 1),
        'age': np.random.randint(18, 70, size=num_customers),
        'gender': np.random.choice(['M', 'F', 'O'], size=num_customers),
        'location': np.random.choice(['Accra', 'Kumasi', 'Tamale', 'Takoradi', 'Cape Coast'], size=num_customers),
        'membership_years': np.random.randint(0, 10, size=num_customers)
    })
    
    # 2. Generate Activity
    activities = []
    start_date = datetime(2025, 1, 1)
    
    for _ in range(num_records):
        cust_id = np.random.randint(1, num_customers + 1)
        timestamp = start_date + timedelta(
            days=np.random.randint(0, 120),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60)
        )
        category = np.random.choice(['Electronics', 'Clothing', 'Home', 'Beauty', 'Books'])
        action = np.random.choice(['view', 'add_to_cart', 'purchase'], p=[0.7, 0.2, 0.1])
        amount = 0
        if action == 'purchase':
            amount = np.round(np.random.uniform(10, 500), 2)
            
        activities.append([cust_id, timestamp, category, action, amount])
        
    activity_df = pd.DataFrame(activities, columns=['customer_id', 'timestamp', 'category', 'action', 'amount'])
    
    # Merge
    df = activity_df.merge(customers, on='customer_id')
    return df

if __name__ == "__main__":
    data = generate_ecommerce_data()
    data.to_csv('ecommerce_data.csv', index=False)
    print("Synthetic dataset generated: ecommerce_data.csv")
