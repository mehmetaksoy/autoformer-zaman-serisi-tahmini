# 🎬 BERT ile IMDB Film Yorumları Duygu Analizi

Bu proje, BERT (Bidirectional Encoder Representations from Transformers) modelini kullanarak IMDB film yorumları üzerinde duygu analizi gerçekleştiren kapsamlı bir makine öğrenmesi uygulamasıdır.

## 📋 Proje Özeti

Bu çalışma, doğal dil işleme (NLP) alanındaki güçlü transformatör mimarilerinden biri olan BERT modelini kullanarak metin sınıflandırma görevini gerçekleştirir. IMDB film yorumları veri seti üzerinde, yorumların pozitif mi yoksa negatif mi olduğunu tahmin eden bir model geliştirilmiştir.

### 🎯 Temel Amaçlar
- Transformatör modellerinin metin verileri üzerindeki etkinliğini pratik bir örnekle göstermek
- Derin öğrenme projesinin tüm adımlarını kapsamlı bir şekilde deneyimlemek
- Hugging Face ekosisteminin modern NLP kütüphanelerinin kullanımını pekiştirmek

## 🏆 Ana Sonuçlar

| Metrik | Değer |
|--------|-------|
| **Test Doğruluğu** | %92.04 |
| **Test F1-Score** | 0.9205 |
| **Test AUC** | 0.9758 |
| **Eğitim Süresi** | ~22.12 dakika |
| **Çıkarım Hızı** | ~3,242 örnek/saniye |

## 📊 Veri Seti

**IMDB Film Yorumları Veri Seti** kullanılmıştır:
- **Kaynak**: Hugging Face Datasets (`load_dataset("imdb")`)
- **Toplam Örnek Sayısı**: 50,000 etiketli yorum
- **Eğitim Seti**: 20,000 örnek (%80)
- **Doğrulama Seti**: 5,000 örnek (%20)
- **Test Seti**: 25,000 örnek
- **Sınıflar**: Pozitif (1) ve Negatif (0) - Dengeli dağılım

## 🔧 Teknik Detaylar

### Model Mimarisi
- **Temel Model**: `bert-base-uncased`
- **Sınıflandırma Katmanı**: 2 sınıf (pozitif/negatif)
- **Tokenizasyon**: WordPiece, max_length=256
- **Parametre Sayısı**: ~110M

### Eğitim Konfigürasyonu
```python
# Ana eğitim parametreleri
num_train_epochs = 3
per_device_train_batch_size = 16
per_device_eval_batch_size = 32
learning_rate = 5e-5
warmup_steps = 500
weight_decay = 0.01
```

### Kütüphane Versiyonları
Proje, aşağıdaki stabil kütüphane kombinasyonu ile geliştirilmiştir:
- **Python**: 3.11.11
- **PyTorch**: 2.5.1+cu124
- **Transformers**: 4.48.3
- **Datasets**: 3.6.0
- **NumPy**: 1.26.4

## 🚀 Kurulum ve Çalıştırma

### 1. Gereksinimler
```bash
pip install torch transformers datasets scikit-learn matplotlib seaborn numpy pandas
```

### 2. Google Colab'da Çalıştırma
1. Notebook dosyasını (`BERT_ile_Metin_Sınıflandırma_(IMDB_Duygu_Analizi)Kopya.ipynb`) Google Colab'a yükleyin
2. GPU Runtime'ını aktifleştirin: `Runtime > Change runtime type > Hardware accelerator > GPU`
3. **Önemli**: İlk hücreyi çalıştırdıktan sonra runtime'ı yeniden başlatın
4. Hücreleri sırasıyla çalıştırın

### 3. Hücre Yapısı
```
Hücre 1: Kütüphane kurulumu (sonrasında runtime restart gerekli)
Hücre 2: Import işlemleri ve versiyon kontrolü
Hücre 3: IMDB veri setinin yüklenmesi
Hücre 4: Veri setinin eğitim/doğrulama/test olarak bölünmesi
Hücre 5: BERT tokenizer yükleme ve tokenizasyon
Hücre 6: BERT modelinin yüklenmesi
Hücre 7: Model eğitimi (epoch bazında değerlendirme)
Hücre 8: Test seti performans değerlendirmesi
Hücre 9: Karmaşıklık matrisi ve ROC eğrisi görselleştirmeleri
Hücre 10: Eğitim kayıp grafikleri
Hücre 11: Çıkarım hızı hesaplamaları
```

## 📈 Performans Analizi

### Epoch Bazında Gelişim
| Epoch | Eğitim Kaybı | Doğrulama Kaybı | Doğrulama Doğruluğu |
|-------|--------------|-----------------|---------------------|
| 1 | 0.3418 | 0.2475 | 0.9098 |
| 2 | 0.1790 | 0.2194 | 0.9182 |
| 3 | 0.0768 | 0.3635 | 0.9188 |

### Test Seti Karmaşıklık Matrisi
```
           Tahmin
Gerçek    Neg    Pos
Neg     11490   1010
Pos       981  11519
```

### Önemli Gözlemler
- Model 2. epoch sonrasında aşırı öğrenme (overfitting) belirtileri göstermiştir
- `load_best_model_at_end=True` ile en iyi performanslı model kullanılmıştır
- Test seti üzerinde yüksek genelleme performansı elde edilmiştir

## 🛠️ Teknik Özellikler

### Metrik Hesaplama
Proje, kapsamlı performans değerlendirmesi için aşağıdaki metrikleri hesaplar:
- **Accuracy**: Genel doğruluk oranı
- **Precision**: Pozitif tahminlerin doğruluk oranı
- **Recall (Sensitivity)**: Pozitif örneklerin yakalanma oranı
- **F1-Score**: Precision ve Recall'ın harmonik ortalaması
- **Specificity**: Negatif örneklerin doğru tanınma oranı
- **AUC**: ROC eğrisi altında kalan alan

### Görselleştirmeler
- Epoch bazında eğitim ve doğrulama kayıp grafikleri
- Doğrulama doğruluk gelişim grafiği
- Test seti karmaşıklık matrisi (heatmap)
- ROC eğrisi ve AUC skoru

## 🔍 Proje Yapısı

```
├── BERT_ile_Metin_Sınıflandırma_(IMDB_Duygu_Analizi)Kopya.ipynb
├── README.md
└── results_epoch_evaluation/
    ├── best_model/           # En iyi model checkpoint'i
    ├── checkpoint-*/         # Epoch bazında checkpoint'ler
    └── runs/                 # TensorBoard logları
```

## 🎓 Öğrenme Çıktıları

Bu proje aşağıdaki konularda pratik deneyim sağlar:
- **BERT modelinin fine-tuning süreci**
- **Hugging Face Transformers ve Datasets kütüphaneleri**
- **PyTorch ve Trainer API kullanımı**
- **Metin ön işleme ve tokenizasyon**
- **Model değerlendirme ve görselleştirme teknikleri**
- **Kütüphane versiyon yönetimi ve hata ayıklama**

## 🚨 Önemli Notlar

### Kütüphane Uyumluluğu
Proje gelişimi sırasında NumPy ve Datasets versiyonları arasında uyumluluk sorunları yaşanmıştır. Bu sorunlar, spesifik versiyonların sabitlenmesi ile çözülmüştür.

### Bellek ve Hesaplama Gereksinimleri
- **GPU**: NVIDIA L4 (Colab Pro) önerilir
- **RAM**: En az 12-16 GB
- **Depolama**: Model checkpoint'leri için ~2-3 GB

## 🔮 Gelişime Açık Alanlar

1. **Model Çeşitlendirmesi**: RoBERTa, DeBERTa gibi diğer transformatör modelleri
2. **Hiperparametre Optimizasyonu**: Learning rate, batch size, epoch sayısı
3. **Tokenizasyon İyileştirmesi**: Farklı max_length değerleri
4. **Regularizasyon**: Dropout, early stopping teknikleri
5. **Ensemble Yöntemleri**: Birden fazla modelin kombinasyonu

## 📝 Lisans

Bu proje eğitim amaçlı geliştirilmiştir ve MIT lisansı altında paylaşılmaktadır.

## 👨‍💻 Katkıda Bulunma

Proje geliştirmelerine katkıda bulunmak için:
1. Repository'yi fork edin
2. Feature branch oluşturun (`git checkout -b feature/YeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/YeniOzellik`)
5. Pull Request oluşturun

## 📞 İletişim

Sorularınız için GitHub Issues bölümünü kullanabilirsiniz.

---

*Bu proje, modern NLP tekniklerinin pratik uygulamasını göstermek ve BERT modelinin güçlü özelliklerini keşfetmek amacıyla geliştirilmiştir.*