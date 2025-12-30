import numpy as np
from multiprocessing import Pool

def parallel_gradient_calculation(X_chunk, error_chunk):
    """
    Bu fonksiyon her bir işlemci (worker) tarafından çalıştırılır.
    Verilen veri parçası üzerindeki gradyanı hesaplar.
    
    Matris Çarpımı (X.T * Error) burada gerçekleşir.
    """
    
    gradient_w = np.dot(X_chunk.T, error_chunk)
    
    gradient_b = np.sum(error_chunk)
    
    return gradient_w, gradient_b

class LinearRegressionModel:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    def fit(self, X, y, n_processors=1):
        """
        Modeli eğitir (Gradient Descent).
        n_processors: 1 ise seri, >1 ise paralel çalışır.
        """
        n_samples, n_features = X.shape
        
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        self.cost_history = []
        
        pool = None
        if n_processors > 1:
            print(f"   -> Paralel Eğitim Başlıyor (İşlemci Sayısı: {n_processors})...")
            pool = Pool(processes=n_processors)
            
            chunk_size = int(np.ceil(n_samples / n_processors))
        else:
            print("   -> Seri Eğitim Başlıyor (Tek İşlemci)...")

        for i in range(self.n_iterations):
            
            y_pred = np.dot(X, self.weights) + self.bias
            
            error = y_pred - y
            
            dw = 0
            db = 0
            
            if n_processors == 1:
                dw = (1 / n_samples) * np.dot(X.T, error)
                db = (1 / n_samples) * np.sum(error)
                
            else:
                chunks = []
                for k in range(n_processors):
                    start = k * chunk_size
                    end = min((k + 1) * chunk_size, n_samples)
                    
                    if start >= n_samples:
                        break
                        
                    X_chunk = X[start:end]
                    err_chunk = error[start:end]
                    chunks.append((X_chunk, err_chunk))
                
                results = pool.starmap(parallel_gradient_calculation, chunks)
                
                total_dw = sum(res[0] for res in results)
                total_db = sum(res[1] for res in results)
                
                dw = (1 / n_samples) * total_dw
                db = (1 / n_samples) * total_db

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if i % 100 == 0:
                cost = (1 / (2 * n_samples)) * np.sum(error ** 2)
                self.cost_history.append(cost)

        if pool:
            pool.close()
            pool.join()
            
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias