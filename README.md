Makine Öğrenimi: Sıfırdan Sınıflandırma (Banknote Authentication)

Bu proje, makine öğreniminin temel algoritmalarından olan Doğrusal Regresyon (Linear Regression) ve Lojistik Regresyon (Logistic Regression) modellerinin, scikit-learn gibi hazır makine öğrenimi kütüphaneleri kullanılmadan, tamamen sıfırdan (from scratch) Python ve Numpy kullanılarak geliştirilmesini içermektedir.

Proje kapsamında modeller, sahte ve gerçek banknotları ayırt etmek için eğitilmiş ve performansları çeşitli metrikler üzerinden karşılaştırılmıştır.

📝 Proje Hakkında

Hazır fonksiyonlar kullanmak yerine, algoritmaların arka planındaki matematiği (Maliyet Fonksiyonları, Gradyan İnişi, Türev Hesaplamaları vb.) koda dökerek makine öğrenimi temellerinin pekiştirilmesi amaçlanmıştır.

Öne Çıkan Özellikler:

Hazır kütüphane (scikit-learn vb.) kullanılmadan sıfırdan model inşası (Nesne Yönelimli Programlama - OOP ile).

Kendi algoritmamızla veriyi Train / Test olarak ayırma (train_test_ayir).

Kendi algoritmamızla Z-Score Standartlaştırması (Feature Scaling).

Gradyan İnişi (Gradient Descent) optimizasyonu.

Eğitim ve Test kayıplarının (loss) iterasyon bazında canlı takibi.

Matplotlib ile Karışıklık Matrisi (Confusion Matrix) ve Öğrenme Eğrileri görselleştirmesi.

📊 Veri Seti

Projede Banknote Authentication Dataset (Sahte Banknot Tespiti Veri Seti) kullanılmıştır. Dalgacık (wavelet) dönüşümü kullanılarak elde edilen istatistiksel özellikleri barındırır. Toplam 1372 adet örnek içermektedir.

Özellikler (Features): Varyans, Çarpıklık (Skewness), Basıklık (Kurtosis), Entropi.

Hedef Sınıf (Target): 0 (Gerçek Banknot) ve 1 (Sahte Banknot).

🚀 Kullanılan Yöntemler

Veri Ön İşleme: Modelin daha iyi yakınsaması (converge) için bağımsız değişkenlere Standartlaştırma uygulanmış ve veri seti %80 Eğitim, %20 Test olarak ayrılmıştır. Veri sızıntısını önlemek için ölçeklendirmede sadece eğitim setinin ortalama ve standart sapması referans alınmıştır.

Doğrusal Regresyon: Ortalama Kare Hatası (MSE) maliyet fonksiyonu ve 0.5 eşik değeri (threshold) kullanılarak iki sınıflı bir sınıflandırıcıya dönüştürülmüştür.

Lojistik Regresyon: Çıktılar Sigmoid fonksiyonundan geçirilerek olasılıksal değerlere dönüştürülmüş ve Çapraz Entropi (Log-Loss) maliyet fonksiyonu ile optimize edilmiştir.

⚙️ Kurulum ve Çalıştırma

Gereksinimler

Projeyi çalıştırmak için Python 3.x ve aşağıdaki kütüphanelerin yüklü olması gerekmektedir:

pip install numpy pandas matplotlib


Projeyi Çalıştırma

Repoyu bilgisayarınıza klonlayın:

git clone https://github.com/kullanici-adiniz/reponuzun-adi.git
cd reponuzun-adi


Python dosyasını çalıştırın:

python deneme.py


⚠️ Önemli Not: Kodu çalıştırmadan önce deneme.py dosyasının 6. satırında bulunan veri seti yolunu (path), veri setinin bilgisayarınızdaki bulunduğu konuma göre (örneğin sadece "data_banknote_authentication.txt" olacak şekilde) güncellemeyi unutmayın.

📈 Sonuçlar ve Performans

Her iki model de test veri seti (274 örnek) üzerinde %98 - %99 aralığında doğruluk (accuracy) elde etmiştir. Öğrenme oranı (learning rate) 0.1 ve iterasyon sayısı 1000 olarak belirlendiğinde algoritmaların başarıyla minimum hata noktasına ulaştığı gözlemlenmiştir.

Script çalıştırıldığında terminalde Doğruluk, Kesinlik, Duyarlılık ve F1-Skoru metrikleri yazdırılmakta; aynı zamanda Öğrenme Eğrileri (Loss) ve Karışıklık Matrisleri (Confusion Matrix) iki ayrı pencerede grafiksel olarak sunulmaktadır.
