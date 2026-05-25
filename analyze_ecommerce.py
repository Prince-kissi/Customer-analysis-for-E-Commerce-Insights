import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. Load and Clean Data
def perform_analysis():
    df = pd.read_csv('ecommerce_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Missing values check
    # In synthetic data there are none, but in real data we'd drop or impute
    df = df.dropna()
    
    # 2. Feature Engineering
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Customer level aggregation (Purchase Patterns)
    customer_stats = df.groupby('customer_id').agg({
        'amount': 'sum',
        'action': lambda x: (x == 'purchase').sum(),
        'age': 'first'
    }).rename(columns={'amount': 'total_spend', 'action': 'total_purchases'})
    
    # Target variable: Will the customer become a "High Value" customer? (Spend > median)
    median_spend = customer_stats['total_spend'].median()
    customer_stats['is_high_value'] = (customer_stats['total_spend'] > median_spend).astype(int)
    
    # 3. Modeling: Predict High Value Customers
    X = customer_stats[['age', 'total_purchases']] # Simplified features
    y = customer_stats['is_high_value']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    # Save Report
    with open('model_results.txt', 'w') as f:
        f.write("Modeling Results: Predicting High Value Customers\n")
        f.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred))

    # 4. Visualizations
    plt.figure(figsize=(10, 6))
    sns.histplot(df[df['amount'] > 0]['amount'], bins=30, kde=True, color='blue')
    plt.title('Distribution of Purchase Amounts')
    plt.savefig('purchase_distribution.png')
    
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='category', hue='action')
    plt.title('Actions by Category')
    plt.savefig('category_actions.png')

    print("Analysis complete. Visualizations and results saved.")

if __name__ == "__main__":
    perform_analysis()
