# Ev Fiyat Tahmini için Paralel Regresyon Analizi

Bu proje, "Paralel Programlama" dersi dönem ödevi kapsamında hazırlanmıştır. Projenin amacı, Lineer Regresyon modelinin eğitim aşamasını (Gradient Descent) paralel programlama teknikleri kullanarak hızlandırmak ve farklı işlemci sayılarındaki (1, 4, 8, 16) performans artışını (speedup) ve verimliliği (efficiency) analiz etmektir.

## 1. Gereksinimler (Prerequisites)

Proje **Python 3.x** dilinde yazılmıştır ve aşağıdaki kütüphanelerin kurulu olmasını gerektirir:

* numpy
* pandas
* scikit-learn

Gerekli kütüphaneleri yüklemek için terminalde şu komutu çalıştırabilirsiniz:

"pip install numpy pandas scikit-learn"


## 2. Dosya Yapısı

* **`project_lib.py`**: Lineer Regresyon modelini içeren ana kütüphane dosyasıdır. Paralel hesaplama (multiprocessing) mantığı ve gradyan hesaplama fonksiyonları burada bulunur.
* **`veri_hazırlayıcı.py`**: Ham veri setini okur, temizler, normalizasyon işlemlerini yapar ve eğitim için hazır hale getirerek `housing_prepared.csv` olarak kaydeder.
* **`main.py`**: Projenin ana çalıştırma dosyasıdır. Modeli 1, 4, 8 ve 16 işlemci ile sırayla eğitir ve sonuçları (Süre, MSE, Hızlanma, Verimlilik) raporlar.
* **`main_heavy_test.py`**: (Opsiyonel) Veri setini yapay olarak çoğaltarak "Stres Testi" uygular. Paralelleştirmenin büyük verilerdeki etkisini daha net görmek için kullanılır.

## 3. Kurulum ve Veri Hazırlama

Projeyi çalıştırmadan önce veri setinin hazırlanması gerekmektedir.

1. Ham veri dosyasının (`1553768847-housing.csv`) proje dizininde olduğundan emin olun.
2. Veri hazırlayıcıyı çalıştırın:

"python veri_hazırlayıcı.py"


*Bu işlem sonucunda dizinde `housing_prepared.csv` dosyası oluşturulacaktır.*

## 4. Çalıştırma Talimatları

Proje, ödevde istenildiği üzere kodun içinde 1 (Seri), 4, 8 ve 16 işlemci konfigürasyonlarını **otomatik olarak** sırayla test edecek şekilde tasarlanmıştır. Tek tek parametre girmenize gerek yoktur.

### Standart Testi Çalıştırma

Ana deneyi başlatmak için şu komutu kullanın:

"python main.py"


### Stres Testini Çalıştırma (Büyük Veri Simülasyonu)

Paralel hesaplamanın performansını daha büyük bir veri yükü altında (yaklaşık 2 milyon satır) gözlemlemek isterseniz:

"python main_heavy_test.py"

*Not: Windows ortamında `multiprocessing` kütüphanesinin doğru çalışması için kodlar `if __name__ == "__main__":` bloğu içerisine alınmıştır. Lütfen kodları doğrudan terminalden çalıştırınız.*

## 5. Beklenen Çıktı

Program çalıştığında sırasıyla her işlemci sayısı için eğitim süresini ve Hata Kareler Ortalamasını (MSE) hesaplar. İşlem bitiminde aşağıdaki gibi bir özet tablo ekrana basılır:

======================================================================
       DENEY SONUÇLARI (Linear Regression)       
======================================================================
İşlemci    | Süre (s)        | Hızlanma   | Verimlilik | MSE       
----------------------------------------------------------------------
1          | 4.2500          | 1.00x      | 1.00       | 0.45123
4          | 1.5000          | 2.83x      | 0.71       | 0.45123
8          | 0.9500          | 4.47x      | 0.56       | 0.45123
16         | 0.8500          | 5.00x      | 0.31       | 0.45123
======================================================================