import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

sutun_isimleri = ["Varyans", "Carpiklik", "Basiklik", "Entropi", "Hedef"]
df = pd.read_csv("C:/Users/ALPEREN/Desktop/Makine/PROJE1/DATASET/data_banknote_authentication.txt", names=sutun_isimleri)

X = df.drop("Hedef", axis=1).values
y = df["Hedef"].values

def train_test_ayir(X, y, test_orani=0.2, random_state=42):
    np.random.seed(random_state)
    indeksler = np.random.permutation(len(X)) 
    test_boyutu = int(len(X) * test_orani)
    test_indeksleri = indeksler[:test_boyutu]
    egitim_indeksleri = indeksler[test_boyutu:]
    return X[egitim_indeksleri], X[test_indeksleri], y[egitim_indeksleri], y[test_indeksleri]

X_train, X_test, y_train, y_test = train_test_ayir(X, y)

def standartlastir(X_train, X_test):
    ortalama = np.mean(X_train, axis=0)
    std_sapma = np.std(X_train, axis=0)
    X_train_olcekli = (X_train - ortalama) / std_sapma
    X_test_olcekli = (X_test - ortalama) / std_sapma 
    return X_train_olcekli, X_test_olcekli

X_train, X_test = standartlastir(X_train, X_test)

class DogrusalRegresyonSiniflandirici:
    def __init__(self, ogrenme_orani=0.01, iterasyon=1000):
        self.lr = ogrenme_orani
        self.iterasyon = iterasyon
        self.agirliklar = None
        self.sapma = None
        self.maliyet_gecmisi = [] 
        self.test_maliyet_gecmisi = [] 

    def fit(self, X, y, X_test=None, y_test=None):
        ornek_sayisi, ozellik_sayisi = X.shape
        self.agirliklar = np.zeros(ozellik_sayisi)
        self.sapma = 0

        for _ in range(self.iterasyon):
            y_tahmin = np.dot(X, self.agirliklar) + self.sapma
            maliyet = (1 / (2 * ornek_sayisi)) * np.sum((y_tahmin - y) ** 2)
            self.maliyet_gecmisi.append(maliyet)
            
            if X_test is not None and y_test is not None:
                y_tahmin_test = np.dot(X_test, self.agirliklar) + self.sapma
                test_maliyet = (1 / (2 * len(y_test))) * np.sum((y_tahmin_test - y_test) ** 2)
                self.test_maliyet_gecmisi.append(test_maliyet)
            
            dw = (1 / ornek_sayisi) * np.dot(X.T, (y_tahmin - y))
            db = (1 / ornek_sayisi) * np.sum(y_tahmin - y)
            self.agirliklar -= self.lr * dw
            self.sapma -= self.lr * db

    def predict(self, X):
        dogrusal_ciktilar = np.dot(X, self.agirliklar) + self.sapma
        return np.where(dogrusal_ciktilar >= 0.5, 1, 0)

class LojistikRegresyon:
    def __init__(self, ogrenme_orani=0.01, iterasyon=1000):
        self.lr = ogrenme_orani
        self.iterasyon = iterasyon
        self.agirliklar = None
        self.sapma = None
        self.maliyet_gecmisi = []
        self.test_maliyet_gecmisi = [] 

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, X_test=None, y_test=None):
        ornek_sayisi, ozellik_sayisi = X.shape
        self.agirliklar = np.zeros(ozellik_sayisi)
        self.sapma = 0
        epsilon = 1e-9

        for _ in range(self.iterasyon):
            y_tahmin = self._sigmoid(np.dot(X, self.agirliklar) + self.sapma)
            maliyet = (-1 / ornek_sayisi) * np.sum(y * np.log(y_tahmin + epsilon) + (1 - y) * np.log(1 - y_tahmin + epsilon))
            self.maliyet_gecmisi.append(maliyet)
            
            if X_test is not None and y_test is not None:
                y_tahmin_test = self._sigmoid(np.dot(X_test, self.agirliklar) + self.sapma)
                test_maliyet = (-1 / len(y_test)) * np.sum(y_test * np.log(y_tahmin_test + epsilon) + (1 - y_test) * np.log(1 - y_tahmin_test + epsilon))
                self.test_maliyet_gecmisi.append(test_maliyet)
            
            dw = (1 / ornek_sayisi) * np.dot(X.T, (y_tahmin - y))
            db = (1 / ornek_sayisi) * np.sum(y_tahmin - y)
            self.agirliklar -= self.lr * dw
            self.sapma -= self.lr * db

    def predict(self, X):
        y_tahmin = self._sigmoid(np.dot(X, self.agirliklar) + self.sapma)
        return np.where(y_tahmin >= 0.5, 1, 0)

model_dogrusal = DogrusalRegresyonSiniflandirici(ogrenme_orani=0.1, iterasyon=1000)
model_dogrusal.fit(X_train, y_train, X_test=X_test, y_test=y_test)
tahminler_dogrusal = model_dogrusal.predict(X_test)

model_lojistik = LojistikRegresyon(ogrenme_orani=0.1, iterasyon=1000)
model_lojistik.fit(X_train, y_train, X_test=X_test, y_test=y_test)
tahminler_lojistik = model_lojistik.predict(X_test)

def performans_degerlendir(y_gercek, y_tahmin, model_adi):
    TP = np.sum((y_gercek == 1) & (y_tahmin == 1))
    TN = np.sum((y_gercek == 0) & (y_tahmin == 0))
    FP = np.sum((y_gercek == 0) & (y_tahmin == 1))
    FN = np.sum((y_gercek == 1) & (y_tahmin == 0))

    accuracy = (TP + TN) / len(y_gercek)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    print(f"\n--- {model_adi} Sonuçları ---")
    print(f"Karışıklık Matrisi: TP={TP}, TN={TN}, FP={FP}, FN={FN}")
    print(f"Doğruluk (Accuracy) : {accuracy:.4f}")
    print(f"Kesinlik (Precision): {precision:.4f}")
    print(f"Duyarlılık (Recall) : {recall:.4f}")
    print(f"F1-Skoru (F1-Score) : {f1_score:.4f}")
    
performans_degerlendir(y_test, tahminler_dogrusal, "Doğrusal Regresyon")
performans_degerlendir(y_test, tahminler_lojistik, "Lojistik Regresyon")

def matris_ciz(y_gercek, y_tahmin, model_adi, ax):
    TP = np.sum((y_gercek == 1) & (y_tahmin == 1))
    TN = np.sum((y_gercek == 0) & (y_tahmin == 0))
    FP = np.sum((y_gercek == 0) & (y_tahmin == 1))
    FN = np.sum((y_gercek == 1) & (y_tahmin == 0))
    
    matris = np.array([[TN, FP], [FN, TP]])
    
    cax = ax.matshow(matris, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matris[i, j]), va='center', ha='center', size=15, color='black' if matris[i, j] < (np.max(matris)/2) else 'white')
    
    ax.set_title(f'{model_adi} Karışıklık Matrisi', pad=20)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Tahmin: 0', 'Tahmin: 1'])
    ax.set_yticklabels(['Gerçek: 0', 'Gerçek: 1'])

# --- BİRİNCİ PENCERE: KAYIP (LOSS) GRAFİKLERİ ---
fig1, ax1 = plt.subplots(1, 2, figsize=(14, 5))
fig1.canvas.manager.set_window_title('Eğitim ve Test Kaybı Grafikleri')

ax1[0].plot(model_dogrusal.maliyet_gecmisi, color='red', label='Eğitim Kaybı')
ax1[0].plot(model_dogrusal.test_maliyet_gecmisi, color='darkred', linestyle='--', label='Test Kaybı')
ax1[0].set_title('Doğrusal Regresyon Eğitimi (MSE)')
ax1[0].set_xlabel('İterasyon')
ax1[0].set_ylabel('Maliyet')
ax1[0].legend()

ax1[1].plot(model_lojistik.maliyet_gecmisi, color='blue', label='Eğitim Kaybı')
ax1[1].plot(model_lojistik.test_maliyet_gecmisi, color='darkblue', linestyle='--', label='Test Kaybı')
ax1[1].set_title('Lojistik Regresyon Eğitimi (Log-Loss)')
ax1[1].set_xlabel('İterasyon')
ax1[1].set_ylabel('Maliyet')
ax1[1].legend()
fig1.tight_layout()

fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.canvas.manager.set_window_title('Karışıklık Matrisleri')

matris_ciz(y_test, tahminler_dogrusal, "Doğrusal", ax2[0])
matris_ciz(y_test, tahminler_lojistik, "Lojistik", ax2[1])
fig2.tight_layout()

# İki pencereyi de aynı anda ekrana getir
plt.show()