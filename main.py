import pandas as pd
import numpy as np
import time
from project_lib import LinearRegressionModel
from sklearn.model_selection import train_test_split

def load_and_split_data(filename):
    df = pd.read_csv(filename)
    
    target_col = 'median_house_value'
    
    X = df.drop(target_col, axis=1).values
    y = df[target_col].values.reshape(-1, 1)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    print("Veri yükleniyor...")
    X_train, X_test, y_train, y_test = load_and_split_data("housing_prepared.csv")
    print(f"Eğitim Verisi: {X_train.shape}, Test Verisi: {X_test.shape}")
    print("-" * 50)

    processor_counts = [1, 4, 8, 16]
    results = []

    LR = 0.1      
    EPOCHS = 1000 

    for p in processor_counts:
        print(f"\n>>> TEST BAŞLIYOR: {p} İşlemci <<<")
        
        model = LinearRegressionModel(learning_rate=LR, n_iterations=EPOCHS)
        
        start_time = time.time()
        
        model.fit(X_train, y_train, n_processors=p)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        predictions = model.predict(X_test)
        mse = np.mean((predictions - y_test) ** 2)
        
        print(f"Tamamlandı. Süre: {elapsed_time:.4f} saniye")
        print(f"Test MSE: {mse:.5f}")
        
        results.append({
            "Processors": p,
            "Time (s)": elapsed_time,
            "MSE": mse
        })

    print("\n" + "="*55)
    print("       DENEY SONUÇLARI (Linear Regression)       ")
    print("="*55)
    print(f"{'İşlemci':<10} | {'Süre (s)':<15} | {'Hızlanma':<10} | {'MSE':<10}")
    print("-" * 55)
    
    base_time = results[0]["Time (s)"] 
    
    for r in results:
        speedup = base_time / r["Time (s)"]
        print(f"{r['Processors']:<10} | {r['Time (s)']:<15.4f} | {speedup:<10.2f}x | {r['MSE']:.5f}")
    print("="*55)