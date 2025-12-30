import pandas as pd
import numpy as np

def prepare_data(input_file, output_file):
    print(f"--- Veri Hazırlama Başladı: {input_file} ---")
    
    df = pd.read_csv(input_file)
    original_count = len(df)
    print(f"Orijinal Veri Sayısı: {original_count}")

    df_clean = df.dropna(subset=['total_bedrooms']).copy()
    dropped_count = original_count - len(df_clean)
    print(f"Silinen Satır Sayısı: {dropped_count}")
    print(f"Kalan Veri Sayısı: {len(df_clean)}")

    feature_cols = ['median_income', 'total_rooms', 'housing_median_age', 'total_bedrooms']
    target_col = 'median_house_value'

    df_selected = df_clean[feature_cols + [target_col]]

    
    means = df_selected.mean()
    stds = df_selected.std()

    df_normalized = (df_selected - means) / stds


    df_normalized.to_csv(output_file, index=False)
    print(f"--- İşlem Tamamlandı. Dosya kaydedildi: {output_file} ---")

    print("\nKontrol (İlk 5 satır):")
    print(df_normalized.head())

if __name__ == "__main__":
    INPUT_CSV = "1553768847-housing.csv"  
    OUTPUT_CSV = "housing_prepared.csv"   
    
    prepare_data(INPUT_CSV, OUTPUT_CSV)