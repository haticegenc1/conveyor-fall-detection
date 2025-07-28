# 📦 Konveyör Bandı Kutu Düşme Tespiti

**YOLOv8 tabanlı gelişmiş görselleştirme ve konfüzyon matrisi analizi ile konveyör bandında kutu düşme tespiti sistemi**

## 📋 Proje Özeti

Bu proje, konveyör bantlarında hareket eden kutuların düşme durumlarını tespit etmek için YOLOv8n modelini kullanır. 4GB RAM için optimize edilmiş, kapsamlı görselleştirmeler ve detaylı performans analizi içerir.

## ✨ Özellikler

- 🎯 **YOLOv8n** ile hızlı ve etkili kutu düşme tespiti
- 💾 **4GB RAM** için optimize edilmiş eğitim parametreleri
- 📊 **Kapsamlı veri seti analizi** ve görselleştirme
- 🔄 **Konfüzyon matrisi** ile detaylı performans analizi
- 📈 **Eğitim süreci izlemi** (loss curves, metrics)
- 🖼️ **Tahmin showcase** ve örnek görüntü grid'i
- 📄 **HTML raporu** ile profesyonel sunum
- 🍎 **macOS M1/M2** Metal Performance Shaders desteği

## 🛠️ Teknolojiler

- **Python 3.8+**
- **YOLOv8** (Ultralytics)
- **PyTorch** 
- **OpenCV**
- **Matplotlib & Seaborn**
- **Scikit-learn**
- **Pandas & NumPy**

### Örnek Kutu Düşme Tespiti
| Normal Hareket | Kutu Düşme Tespiti |
|----------------|-------------------|
|<img width="1440" height="820" alt="image" src="https://github.com/user-attachments/assets/49c8c417-bc84-4882-9469-c8d55b99520b" />|<img width="1440" height="820" alt="image" src="https://github.com/user-attachments/assets/8a653ee6-d0f0-43ec-8e8e-d140d1515489" /> 
| ✅ Düzgün konveyör hareketi | ❌ Tespit edilen kutu düşmesi |


## 📊 Çıktılar ve Görselleştirmeler

<img width="2066" height="840" alt="image" src="https://github.com/user-attachments/assets/3130d69d-0d7a-4bc6-a6da-3fc0cb345adb" />
<img width="5970" height="2585" alt="image" src="https://github.com/user-attachments/assets/71709e34-ff18-4f6a-a5ea-6a7e142d3b04" />
<img width="2878" height="1730" alt="image" src="https://github.com/user-attachments/assets/38d576ba-f590-485d-a744-855df04083d0" />
<img width="4648" height="1768" alt="image" src="https://github.com/user-attachments/assets/88eba032-9c6a-475d-9891-e698509763f2" />
<img width="4469" height="1779" alt="image" src="https://github.com/user-attachments/assets/da674105-ce2f-4605-9478-55be4fb61b6a" />


## 🎯 Model Performans Metrikleri

Sistem şu metrikleri hesaplar ve görselleştirir:

- **mAP50** - Mean Average Precision (IoU=0.5)
- **mAP50-95** - Mean Average Precision (IoU=0.5:0.95)
- **Precision** - Kesinlik
- **Recall** - Duyarlılık  
- **F1-Score** - Harmonic mean
- **Accuracy** - Doğruluk
- **Specificity** - Özgüllük

## ⚙️ Konfigürasyon

### Eğitim Parametreleri
```python
train_args = {
    'epochs': 20,           # Eğitim epoch sayısı
    'imgsz': 416,          # Görüntü boyutu
    'batch': 4,            # Batch size (4GB RAM için)
    'device': 'mps',       # macOS için 'mps', Windows/Linux için 'cuda' veya 'cpu'
    'patience': 15,        # Early stopping patience
    'workers': 2,          # Data loader worker sayısı
    'amp': False,          # Automatic Mixed Precision
    'cache': False,        # Görüntü cache (RAM tasarrufu için False)
}
```

### Data Augmentation
```python
augmentation_args = {
    'hsv_h': 0.01,         # Hue augmentation
    'hsv_s': 0.3,          # Saturation augmentation  
    'hsv_v': 0.2,          # Value augmentation
    'degrees': 3,          # Rotation range
    'translate': 0.03,     # Translation range
    'scale': 0.2,          # Scale range
    'flipud': 0.0,         # Vertical flip probability
    'fliplr': 0.5,         # Horizontal flip probability
    'mosaic': 0.8,         # Mosaic augmentation probability
}
```

## 💡 Optimizasyon İpuçları

### 4GB RAM için:
- `batch_size=4` 
- `cache=False` 
- `workers=2` 
- `amp=False` 
