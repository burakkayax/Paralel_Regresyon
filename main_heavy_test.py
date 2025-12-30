
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
    print("="*60)
    print("   STRES TESTİ MODU: Veri Seti Yapay Olarak Büyütülüyor")
    print("="*60)


    print("Orijinal Veri yükleniyor...")
    X_train_orig, X_test, y_train_orig, y_test = load_and_split_data("housing_prepared.csv")
    print(f"Orijinal Eğitim Boyutu: {X_train_orig.shape}")

    REPLICATION_FACTOR = 100
    print(f"\nVeri {REPLICATION_FACTOR} katına çıkarılıyor...")
    
    X_train = np.tile(X_train_orig, (REPLICATION_FACTOR, 1))
    y_train = np.tile(y_train_orig, (REPLICATION_FACTOR, 1))
    
    print(f"YENİ Eğitim Boyutu: {X_train.shape} (Yaklaşık {X_train.shape[0]/1_000_000:.1f} Milyon Satır)")
    print("-" * 60)

    processor_counts = [1, 4, 8, 16]
    results = []

    LR = 0.1       
    EPOCHS = 100   

    for p in processor_counts:
        print(f"\n>>> STRES TESTİ BAŞLIYOR: {p} İşlemci <<<")
        
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

    print("\n" + "="*60)
    print("       STRES TESTİ SONUÇLARI (Büyük Veri)       ")
    print("="*60)
    print(f"{'İşlemci':<10} | {'Süre (s)':<15} | {'Hızlanma':<10} | {'MSE':<10}")
    print("-" * 60)
    
    base_time = results[0]["Time (s)"] # 1 İşlemci süresi referanstır
    
    for r in results:
        speedup = base_time / r["Time (s)"]
        print(f"{r['Processors']:<10} | {r['Time (s)']:<15.4f} | {speedup:<10.2f}x | {r['MSE']:.5f}")
    print("="*60)