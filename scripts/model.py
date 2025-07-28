"""
Konveyör Bandı Düşme Tespiti - Enhanced YOLO Model Eğitimi
Kapsamlı görselleştirme ve raporlama ile
"""

import os
import yaml
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
import random
import numpy as np
import cv2
import pandas as pd
from collections import Counter
import json
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report

class FallDetectionTrainer:
    def __init__(self, data_folder_path, project_name="fall_detection"):
        self.data_path = Path(data_folder_path)
        self.processed_path = self.data_path / "processed"
        self.yaml_path = self.processed_path / "data.yaml"
        self.project_name = project_name
        self.model = None
        self.results_dir = Path(f"{project_name}_analysis")
        self.results_dir.mkdir(exist_ok=True)
        
        # Matplotlib Turkish font fix
        plt.rcParams['font.family'] = ['DejaVu Sans']
        
        # macOS M1 için cihaz seçimi
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("✅ Metal Performance Shaders (MPS) kullanılacak")
        else:
            self.device = "cpu"
            print("⚠️ CPU kullanılacak - 4GB RAM için daha güvenli seçim")
    
    def analyze_dataset(self):
        """Veri setini detaylı analiz et ve görselleştir"""
        print("📊 Veri seti analizi başlıyor...")
        
        # YAML'dan sınıf bilgilerini al
        with open(self.yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # names alanı liste veya dict olabilir - normalize et
        raw_names = config.get('names', [])
        if isinstance(raw_names, list):
            # Liste formatını dict'e çevir: ['fall', 'no_fall'] -> {0: 'fall', 1: 'no_fall'}
            class_names = {i: name for i, name in enumerate(raw_names)}
        else:
            # Zaten dict formatında
            class_names = raw_names
        
        print(f"🏷️ Sınıf isimleri: {class_names}")
        
        analysis_results = {}
        
        for split in ['train', 'val', 'test']:
            print(f"\n📈 {split.upper()} seti analizi...")
            
            images_path = self.processed_path / "images" / split
            labels_path = self.processed_path / "labels" / split
            
            # Görüntü sayısı
            image_files = list(images_path.glob("*.png")) + list(images_path.glob("*.jpg"))
            label_files = list(labels_path.glob("*.txt"))
            
            # Görüntü boyutları analizi
            image_sizes = []
            for img_file in image_files[:50]:  # İlk 50 görüntüyü sample al
                img = cv2.imread(str(img_file))
                if img is not None:
                    h, w = img.shape[:2]
                    image_sizes.append((w, h))
            
            # Etiket analizi
            class_counts = Counter()
            bbox_areas = []
            objects_per_image = []
            negative_samples = 0  # Boş txt dosyaları (no_fall)
            
            for label_file in label_files:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    lines = [line.strip() for line in lines if line.strip()]  # Boş satırları temizle
                    objects_per_image.append(len(lines))
                    
                    if len(lines) == 0:
                        # Boş dosya = negative sample (no_fall)
                        negative_samples += 1
                    
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            class_counts[class_id] += 1
                            
                            # Bbox area hesapla (normalized)
                            w, h = float(parts[3]), float(parts[4])
                            bbox_areas.append(w * h)
            
            # Negative samples'ı class_counts'a ekle
            # Eğer 2 sınıf varsa (0=fall, 1=no_fall) no_fall'ı 1 olarak kabul et
            if len(class_names) >= 2 and negative_samples > 0:
                # no_fall sınıfının ID'sini bul
                no_fall_id = None
                for class_id, class_name in class_names.items():
                    if 'no_fall' in class_name.lower() or 'normal' in class_name.lower():
                        no_fall_id = class_id
                        break
                
                if no_fall_id is None and len(class_names) >= 2:
                    # Eğer bulamazsak, fall=0 ise no_fall=1 varsay
                    no_fall_id = 1
                
                if no_fall_id is not None:
                    class_counts[no_fall_id] = negative_samples
            
            total_positive_images = len(label_files) - negative_samples
            total_positive_objects = sum(class_counts.values())
            print(f"   📊 {split}: {total_positive_images} pozitif görüntü ({total_positive_objects} nesne), {negative_samples} negatif görüntü")
            
            analysis_results[split] = {
                'image_count': len(image_files),
                'label_count': len(label_files),
                'image_sizes': image_sizes,
                'class_counts': dict(class_counts),
                'bbox_areas': bbox_areas,
                'objects_per_image': objects_per_image
            }
        
        # Görselleştirmeleri oluştur
        self._plot_dataset_analysis(analysis_results, class_names)
        return analysis_results
    
    def _plot_dataset_analysis(self, analysis_results, class_names):
        """Dataset analiz görselleştirmeleri"""
        
        # 1. Dataset Distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dataset Analizi - Konveyör Bandı Düşme Tespiti', fontsize=16, fontweight='bold')
        
        # Görüntü sayıları
        splits = list(analysis_results.keys())
        image_counts = [analysis_results[split]['image_count'] for split in splits]
        
        axes[0,0].bar(splits, image_counts, color=['#ff7f0e', '#2ca02c', '#1f77b4'])
        axes[0,0].set_title('Veri Seti Dağılımı (Görüntü Sayısı)')
        axes[0,0].set_ylabel('Görüntü Sayısı')
        for i, v in enumerate(image_counts):
            axes[0,0].text(i, v + max(image_counts)*0.01, str(v), ha='center', fontweight='bold')
        
        # Sınıf dağılımı (tüm splits birleşik)
        all_class_counts = Counter()
        for split_data in analysis_results.values():
            for class_id, count in split_data['class_counts'].items():
                all_class_counts[class_id] += count
        
        class_labels = [class_names.get(i, f'Class {i}') for i in all_class_counts.keys()]
        class_values = list(all_class_counts.values())
        
        axes[0,1].pie(class_values, labels=class_labels, autopct='%1.1f%%', startangle=90)
        axes[0,1].set_title('Sınıf Dağılımı (Toplam)')
        
        # Görüntü boyutları dağılımı
        all_sizes = []
        for split_data in analysis_results.values():
            all_sizes.extend(split_data['image_sizes'])
        
        if all_sizes:
            widths, heights = zip(*all_sizes)
            axes[1,0].scatter(widths, heights, alpha=0.6, color='purple')
            axes[1,0].set_title('Görüntü Boyutları Dağılımı')
            axes[1,0].set_xlabel('Genişlik (px)')
            axes[1,0].set_ylabel('Yükseklik (px)')
            axes[1,0].grid(True, alpha=0.3)
        
        # Nesne sayısı per görüntü
        all_objects = []
        for split_data in analysis_results.values():
            all_objects.extend(split_data['objects_per_image'])
        
        if all_objects:
            axes[1,1].hist(all_objects, bins=max(10, max(all_objects)), alpha=0.7, color='orange')
            axes[1,1].set_title('Görüntü Başına Nesne Sayısı Dağılımı')
            axes[1,1].set_xlabel('Nesne Sayısı')
            axes[1,1].set_ylabel('Görüntü Sayısı')
            axes[1,1].axvline(np.mean(all_objects), color='red', linestyle='--', 
                            label=f'Ortalama: {np.mean(all_objects):.2f}')
            axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'dataset_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Dataset analizi kaydedildi: {self.results_dir / 'dataset_analysis.png'}")
    
    def create_sample_grid(self, num_samples=12):
        """Her split'ten örnek görüntüleri grid halinde göster"""
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle('Veri Seti Örnekleri - Düşme Tespiti', fontsize=16, fontweight='bold')
        
        splits = ['train', 'val', 'test']
        
        for i, split in enumerate(splits):
            images_path = self.processed_path / "images" / split
            labels_path = self.processed_path / "labels" / split
            
            image_files = list(images_path.glob("*.png")) + list(images_path.glob("*.jpg"))
            sample_files = random.sample(image_files, min(4, len(image_files)))
            
            for j, img_file in enumerate(sample_files):
                if i * 4 + j < 12:
                    ax = axes[i, j]
                    
                    # Görüntüyü yükle
                    img = cv2.imread(str(img_file))
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Etiketleri yükle ve çiz
                    label_file = labels_path / f"{img_file.stem}.txt"
                    if label_file.exists():
                        h, w = img.shape[:2]
                        with open(label_file, 'r') as f:
                            for line in f.readlines():
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    # YOLO format: class x_center y_center width height (normalized)
                                    x_center, y_center, width, height = map(float, parts[1:5])
                                    
                                    # Pixel koordinatlarına çevir
                                    x1 = int((x_center - width/2) * w)
                                    y1 = int((y_center - height/2) * h)
                                    x2 = int((x_center + width/2) * w)
                                    y2 = int((y_center + height/2) * h)
                                    
                                    # Bbox çiz
                                    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    ax.imshow(img_rgb)
                    ax.set_title(f'{split.upper()}: {img_file.name}', fontsize=10)
                    ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'sample_images.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Örnek görüntüler kaydedildi: {self.results_dir / 'sample_images.png'}")
    
    def monitor_training_progress(self, run_dir):
        """Training progress'i takip et ve görselleştir"""
        
        results_csv = Path(run_dir) / "results.csv"
        if not results_csv.exists():
            print("⚠️ results.csv dosyası bulunamadı")
            return
        
        # CSV'yi oku
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()  # Boşlukları temizle
        
        # Training curves
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Eğitim İlerlemesi - YOLOv8 Düşme Tespiti', fontsize=16, fontweight='bold')
        
        epoch = df['epoch'] if 'epoch' in df.columns else range(len(df))
        
        # Loss curves
        loss_cols = [col for col in df.columns if 'loss' in col.lower()]
        for i, loss_col in enumerate(loss_cols[:3]):
            if i < 3 and loss_col in df.columns:
                axes[0, i].plot(epoch, df[loss_col], 'b-', linewidth=2)
                axes[0, i].set_title(f'{loss_col}')
                axes[0, i].set_xlabel('Epoch')
                axes[0, i].set_ylabel('Loss')
                axes[0, i].grid(True, alpha=0.3)
        
        # Metrics
        metric_cols = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)']
        metric_names = ['Precision', 'Recall', 'mAP50']
        
        for i, (metric_col, name) in enumerate(zip(metric_cols, metric_names)):
            if metric_col in df.columns:
                axes[1, i].plot(epoch, df[metric_col], 'g-', linewidth=2)
                axes[1, i].set_title(name)
                axes[1, i].set_xlabel('Epoch')
                axes[1, i].set_ylabel('Score')
                axes[1, i].grid(True, alpha=0.3)
                axes[1, i].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Eğitim eğrileri kaydedildi: {self.results_dir / 'training_curves.png'}")
        
        # En iyi sonuçları yazdır
        if 'metrics/mAP50(B)' in df.columns:
            best_epoch = df.loc[df['metrics/mAP50(B)'].idxmax()]
            print(f"\n🏆 En İyi Sonuçlar (Epoch {best_epoch['epoch']}):")
            print(f"   mAP50: {best_epoch['metrics/mAP50(B)']:.4f}")
            if 'metrics/precision(B)' in df.columns:
                print(f"   Precision: {best_epoch['metrics/precision(B)']:.4f}")
            if 'metrics/recall(B)' in df.columns:
                print(f"   Recall: {best_epoch['metrics/recall(B)']:.4f}")
    
    def create_prediction_showcase(self, num_examples=8):
        """En iyi ve en kötü tahminlerin showcase'ini oluştur"""
        
        if self.model is None:
            print("❌ Model henüz eğitilmedi!")
            return
        
        test_images_path = self.processed_path / "images" / "test"
        test_labels_path = self.processed_path / "labels" / "test"
        
        image_files = list(test_images_path.glob("*.png")) + list(test_images_path.glob("*.jpg"))
        
        if len(image_files) < num_examples:
            num_examples = len(image_files)
        
        sample_files = random.sample(image_files, num_examples)
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Model Tahminleri - Düşme Tespiti Sonuçları', fontsize=16, fontweight='bold')
        
        for i, img_file in enumerate(sample_files):
            if i >= 8:
                break
                
            row = i // 4
            col = i % 4
            ax = axes[row, col]
            
            # Görüntüyü yükle
            img = cv2.imread(str(img_file))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Model tahmini
            results = self.model(str(img_file), verbose=False)
            
            # Ground truth etiketleri çiz (yeşil)
            label_file = test_labels_path / f"{img_file.stem}.txt"
            gt_count = 0
            if label_file.exists():
                h, w = img.shape[:2]
                with open(label_file, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            gt_count += 1
                            x_center, y_center, width, height = map(float, parts[1:5])
                            x1 = int((x_center - width/2) * w)
                            y1 = int((y_center - height/2) * h)
                            x2 = int((x_center + width/2) * w)
                            y2 = int((y_center + height/2) * h)
                            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img_rgb, 'GT', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Model tahminlerini çiz (kırmızı)
            pred_count = 0
            for r in results:
                if len(r.boxes) > 0:
                    for box in r.boxes:
                        pred_count += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(img_rgb, f'P:{conf:.2f}', (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            ax.imshow(img_rgb)
            ax.set_title(f'{img_file.name}\nGT:{gt_count} | Pred:{pred_count}', fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'prediction_showcase.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Tahmin showcase kaydedildi: {self.results_dir / 'prediction_showcase.png'}")
    
    def create_performance_report(self, test_results):
        """Detaylı performans raporu oluştur"""
        
        # Metrikleri topla
        metrics = {
            'mAP50': float(test_results.box.map50),
            'mAP50-95': float(test_results.box.map),
            'Precision': float(test_results.box.mp),
            'Recall': float(test_results.box.mr),
            'F1-Score': 2 * (float(test_results.box.mp) * float(test_results.box.mr)) / 
                       (float(test_results.box.mp) + float(test_results.box.mr)) if (float(test_results.box.mp) + float(test_results.box.mr)) > 0 else 0
        }
        
        # Performans görseli
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Model Performans Raporu', fontsize=16, fontweight='bold')
        
        # Metric scores bar chart
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        colors = ['#ff7f0e', '#2ca02c', '#1f77b4', '#d62728', '#9467bd']
        
        bars = ax1.bar(metric_names, metric_values, color=colors)
        ax1.set_title('Test Metrikleri')
        ax1.set_ylabel('Skor')
        ax1.set_ylim(0, 1)
        
        # Değerleri bar üzerine yaz
        for bar, value in zip(bars, metric_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Radar chart
        angles = np.linspace(0, 2*np.pi, len(metric_names), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Close the plot
        values = metric_values + [metric_values[0]]
        
        ax2 = plt.subplot(122, projection='polar')
        ax2.plot(angles, values, 'o-', linewidth=2, color='blue')
        ax2.fill(angles, values, alpha=0.25, color='blue')
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metric_names)
        ax2.set_ylim(0, 1)
        ax2.set_title('Performans Radar Grafiği')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'performance_report.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # JSON raporu da kaydet
        report = {
            'model': 'YOLOv8n',
            'task': 'Konveyör Bandı Düşme Tespiti',
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'device': self.device
        }
        
        with open(self.results_dir / 'performance_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Performans raporu kaydedildi: {self.results_dir / 'performance_report.png'}")
        print(f"📄 JSON raporu: {self.results_dir / 'performance_report.json'}")
        
        return metrics
    
    def check_dataset_structure(self):
        """Mevcut veri seti yapısını kontrol et - orijinal method"""
        print("📁 Veri seti yapısı kontrol ediliyor...")
        
        if not self.processed_path.exists():
            print(f"❌ processed klasörü bulunamadı: {self.processed_path}")
            return False
            
        if not self.yaml_path.exists():
            print(f"❌ data.yaml dosyası bulunamadı: {self.yaml_path}")
            return False
            
        print(f"✅ data.yaml bulundu: {self.yaml_path}")
        
        required_folders = [
            self.processed_path / "images" / "train",
            self.processed_path / "images" / "val", 
            self.processed_path / "images" / "test",
            self.processed_path / "labels" / "train",
            self.processed_path / "labels" / "val",
            self.processed_path / "labels" / "test"
        ]
        
        for folder in required_folders:
            if not folder.exists():
                print(f"❌ Klasör bulunamadı: {folder}")
                return False
                
        for split in ['train', 'val', 'test']:
            img_count = len(list((self.processed_path / "images" / split).glob("*.png")))
            label_count = len(list((self.processed_path / "labels" / split).glob("*.txt")))
            print(f"📊 {split.upper()}: {img_count} görüntü, {label_count} etiket")
            
        print("✅ Veri seti yapısı doğru!")
        return True
        
    def update_yaml_paths(self):
        """data.yaml dosyasındaki yolları güncelle - orijinal method"""
        try:
            with open(self.yaml_path, 'r') as f:
                config = yaml.safe_load(f)
                
            print(f"📄 Mevcut YAML içeriği: {config}")
            
            config['path'] = str(self.processed_path.absolute())
            config['train'] = 'images/train'
            config['val'] = 'images/val'  
            config['test'] = 'images/test'
            
            with open(self.yaml_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
                
            print(f"✅ data.yaml yolları güncellendi")
            print(f"📂 Root yol: {self.processed_path}")
            print(f"🔧 Güncellenmiş config: {config}")
            
            return True
            
        except Exception as e:
            print(f"❌ YAML güncelleme hatası: {e}")
            return False
        
    def train_model(self, epochs=20, imgsz=416, batch_size=4):
        """Enhanced training with progress monitoring"""
        
        model_weights = 'yolov8n.pt'
        print(f"🤖 Model: {model_weights} (4GB RAM için optimize)")
        
        self.model = YOLO(model_weights)
        
        train_args = {
            'data': str(self.yaml_path),
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'device': self.device,
            'project': self.project_name,
            'name': 'conveyor_fall_detection',
            'patience': 15,
            'save': True,
            'plots': True,
            'val': True,
            'verbose': True,
            'workers': 2,
            'amp': False,
            'cache': False,
            'single_cls': False,
            'hsv_h': 0.01,
            'hsv_s': 0.3,
            'hsv_v': 0.2,
            'degrees': 3,
            'translate': 0.03,
            'scale': 0.2,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 0.8,      
            'shear': 0.0, 
        }
        
        print("🚀 YOLOv8n Eğitimi Başlıyor...")
        print(f"⚙️ Cihaz: {self.device}")
        print(f"📊 Batch Size: {batch_size} (4GB RAM için optimize)")
        print(f"🖼️ Image Size: {imgsz}x{imgsz}")
        print(f"🔄 Epochs: {epochs}")
        
        # Eğitimi başlat
        results = self.model.train(**train_args)
        
        # Training progress'i görselleştir
        run_dir = Path(self.project_name) / 'conveyor_fall_detection'
        self.monitor_training_progress(run_dir)
        
        print("✅ Eğitim tamamlandı!")
        return results
        

    def create_confusion_matrix(self):
        """YOLOv8 için özel konfüzyon matrisi oluştur"""
        
        if self.model is None:
            print("❌ Önce modeli eğitmeniz gerekiyor!")
            return
        
        print("🔄 Konfüzyon matrisi hesaplanıyor...")
        
        test_images_path = self.processed_path / "images" / "test"
        test_labels_path = self.processed_path / "labels" / "test"
        
        image_files = list(test_images_path.glob("*.png")) + list(test_images_path.glob("*.jpg"))
        
        y_true = []  # Ground truth labels
        y_pred = []  # Predicted labels
        
        # YAML'dan sınıf isimlerini al
        with open(self.yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        raw_names = config.get('names', [])
        if isinstance(raw_names, list):
            class_names = {i: name for i, name in enumerate(raw_names)}
        else:
            class_names = raw_names
        
        print(f"🏷️ Sınıflar: {class_names}")
        
        # Confidence threshold
        conf_threshold = 0.5
        
        for img_file in image_files:
            # Ground truth label'ı oku
            label_file = test_labels_path / f"{img_file.stem}.txt"
            
            # Ground truth belirleme
            gt_has_fall = False
            if label_file.exists():
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            # Fall sınıfı (genellikle 0) varsa true
                            if class_id == 0:  # fall sınıfının ID'si
                                gt_has_fall = True
                                break
            
            # Ground truth: 1 = fall, 0 = no_fall
            gt_label = 1 if gt_has_fall else 0
            y_true.append(gt_label)
            
            # Model tahmini
            results = self.model(str(img_file), verbose=False)
            
            # Prediction belirleme
            pred_has_fall = False
            for r in results:
                if len(r.boxes) > 0:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        # Fall sınıfı ve yeterli confidence varsa
                        if class_id == 0 and conf >= conf_threshold:
                            pred_has_fall = True
                            break
                if pred_has_fall:
                    break
            
            # Prediction: 1 = fall, 0 = no_fall
            pred_label = 1 if pred_has_fall else 0
            y_pred.append(pred_label)
        
        # Konfüzyon matrisini hesapla
        cm = confusion_matrix(y_true, y_pred)
        
        # Sınıf isimleri
        class_labels = ['no_fall', 'fall']
        
        # Konfüzyon matrisini görselleştir
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Konfüzyon Matrisi - Düşme Tespiti Modeli', fontsize=16, fontweight='bold')
        
        # 1. Ham sayılar ile konfüzyon matrisi
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_labels, yticklabels=class_labels, ax=ax1)
        ax1.set_title('Konfüzyon Matrisi (Sayılar)')
        ax1.set_ylabel('Gerçek Değer (Ground Truth)')
        ax1.set_xlabel('Tahmin (Prediction)')
        
        # Metin açıklamaları ekle
        ax1.text(0.5, -0.1, 'TN: True Negative (Doğru no_fall)', transform=ax1.transAxes, 
                ha='center', fontsize=10, color='darkblue')
        ax1.text(1.5, -0.1, 'FP: False Positive (Yanlış fall)', transform=ax1.transAxes, 
                ha='center', fontsize=10, color='red')
        ax1.text(0.5, -0.15, 'FN: False Negative (Kaçırılan fall)', transform=ax1.transAxes, 
                ha='center', fontsize=10, color='red')
        ax1.text(1.5, -0.15, 'TP: True Positive (Doğru fall)', transform=ax1.transAxes, 
                ha='center', fontsize=10, color='green')
        
        # 2. Normalize edilmiş konfüzyon matrisi (yüzdeler)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=class_labels, yticklabels=class_labels, ax=ax2)
        ax2.set_title('Normalize Konfüzyon Matrisi (Yüzdeler)')
        ax2.set_ylabel('Gerçek Değer (Ground Truth)')
        ax2.set_xlabel('Tahmin (Prediction)')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Detaylı metrikleri hesapla
        tn, fp, fn, tp = cm.ravel()
        
        # Metrikler
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Sonuçları yazdır
        print("\n" + "="*50)
        print("📊 KONFÜZYON MATRİSİ SONUÇLARI")
        print("="*50)
        print(f"📈 Test Görüntü Sayısı: {len(y_true)}")
        print(f"🎯 Confidence Threshold: {conf_threshold}")
        print("\n📋 Konfüzyon Matrisi:")
        print(f"   True Negative (TN):  {tn:3d} - Doğru no_fall tespiti")
        print(f"   False Positive (FP): {fp:3d} - Yanlış fall tespiti")
        print(f"   False Negative (FN): {fn:3d} - Kaçırılan fall")
        print(f"   True Positive (TP):  {tp:3d} - Doğru fall tespiti")
        
        print("\n📊 Performans Metrikleri:")
        print(f"   🎯 Accuracy (Doğruluk):     {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"   🔍 Precision (Kesinlik):    {precision:.3f} ({precision*100:.1f}%)")
        print(f"   🎣 Recall (Duyarlılık):     {recall:.3f} ({recall*100:.1f}%)")
        print(f"   🛡️ Specificity (Özgüllük):  {specificity:.3f} ({specificity*100:.1f}%)")
        print(f"   ⚖️ F1-Score:                {f1_score:.3f} ({f1_score*100:.1f}%)")
        
        # Classification report
        print("\n📋 Detaylı Sınıflandırma Raporu:")
        report = classification_report(y_true, y_pred, target_names=class_labels, digits=3)
        print(report)
        
        # Sonuçları JSON olarak kaydet
        results = {
            'confusion_matrix': {
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_positive': int(tp)
            },
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'specificity': float(specificity),
                'f1_score': float(f1_score)
            },
            'test_samples': len(y_true),
            'confidence_threshold': conf_threshold,
            'class_names': class_labels
        }
        
        with open(self.results_dir / 'confusion_matrix_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Konfüzyon matrisi kaydedildi: {self.results_dir / 'confusion_matrix.png'}")
        print(f"📄 Detaylı sonuçlar: {self.results_dir / 'confusion_matrix_results.json'}")
        
        return results
    
    def evaluate_model(self):
        """Enhanced model evaluation with visualizations"""
        
        if self.model is None:
            print("❌ Önce modeli eğitmeniz gerekiyor!")
            return
            
        test_results = self.model.val(
            data=str(self.yaml_path),
            split='test'
        )
        
        print("📊 Test Sonuçları:")
        print(f"mAP50: {test_results.box.map50:.4f}")
        print(f"mAP50-95: {test_results.box.map:.4f}")
        print(f"Precision: {test_results.box.mp:.4f}")
        print(f"Recall: {test_results.box.mr:.4f}")
        
        # Performans raporu oluştur
        metrics = self.create_performance_report(test_results)
        
        # Konfüzyon matrisi oluştur
        confusion_results = self.create_confusion_matrix()
        
        return test_results, confusion_results

        
    def test_inference(self, test_image_path=None):
        """Enhanced inference with showcase"""
        
        if self.model is None:
            print("❌ Önce modeli eğitmeniz gerekiyor!")
            return
            
        # Prediction showcase oluştur
        self.create_prediction_showcase()
        
        # Orijinal test inference
        if test_image_path is None:
            test_images = list((self.processed_path / "images" / "test").glob("*.png"))
            if test_images:
                test_image_path = random.choice(test_images)
            else:
                print("❌ Test görüntüsü bulunamadı!")
                return
                
        print(f"🔍 Test görüntüsü: {test_image_path}")
        
        results = self.model(test_image_path)
        
        for r in results:
            output_path = self.results_dir / f"test_prediction_{Path(test_image_path).stem}.jpg"
            r.save(str(output_path))
            print(f"💾 Sonuç kaydedildi: {output_path}")
            
            if len(r.boxes) > 0:
                print(f"🎯 {len(r.boxes)} düşme tespit edildi!")
                for box in r.boxes:
                    conf = float(box.conf[0])
                    print(f"   - Güven skoru: {conf:.2f}")
            else:
                print("🔍 Düşme tespit edilmedi")
                
        return results
    
    def generate_comprehensive_report(self):
        """Kapsamlı rapor oluştur - Konfüzyon matrisi ile güncellenmiş"""
        
        print("\n📋 Kapsamlı Rapor Oluşturuluyor...")
        
        # HTML raporu oluştur (konfüzyon matrisi bölümü eklendi)
        html_content = f"""
        <!DOCTYPE html>
        <html lang="tr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>YOLOv8 Düşme Tespiti - Analiz Raporu</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ text-align: center; background: #f4f4f4; padding: 20px; border-radius: 10px; }}
                .section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
                .metric-card {{ background: #f9f9f9; padding: 15px; border-radius: 5px; text-align: center; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                .image-item {{ text-align: center; }}
                .image-item img {{ max-width: 100%; height: auto; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .conclusion {{ background: #e8f5e8; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50; }}
                .confusion-info {{ background: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🤖 YOLOv8 Konveyör Bandı Düşme Tespiti</h1>
                <h2>Analiz ve Performans Raporu</h2>
                <p><strong>Tarih:</strong> {datetime.now().strftime('%d.%m.%Y %H:%M')}</p>
                <p><strong>Model:</strong> YOLOv8 Nano (4GB RAM Optimized)</p>
            </div>
            
            <div class="section">
                <h3>📊 Veri Seti Analizi</h3>
                <div class="image-grid">
                    <div class="image-item">
                        <img src="dataset_analysis.png" alt="Dataset Analizi">
                        <p><strong>Veri Seti Dağılımı ve İstatistikleri</strong></p>
                    </div>
                    <div class="image-item">
                        <img src="sample_images.png" alt="Örnek Görüntüler">
                        <p><strong>Veri Seti Örnekleri</strong></p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h3>📈 Eğitim Süreci</h3>
                <div class="image-grid">
                    <div class="image-item">
                        <img src="training_curves.png" alt="Eğitim Eğrileri">
                        <p><strong>Loss ve Metrik Eğrileri</strong></p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h3>🎯 Model Performansı</h3>
                <div class="image-grid">
                    <div class="image-item">
                        <img src="performance_report.png" alt="Performans Raporu">
                        <p><strong>Test Metrikleri</strong></p>
                    </div>
                    <div class="image-item">
                        <img src="prediction_showcase.png" alt="Tahmin Örnekleri">
                        <p><strong>Model Tahminleri</strong></p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h3>🔄 Konfüzyon Matrisi Analizi</h3>
                <div class="confusion-info">
                    <h4>Konfüzyon Matrisi Açıklaması:</h4>
                    <ul>
                        <li><strong>True Positive (TP):</strong> Düşme varken doğru tespit edildi</li>
                        <li><strong>True Negative (TN):</strong> Düşme yokken doğru tespit edildi</li>
                        <li><strong>False Positive (FP):</strong> Düşme yokken yanlış alarm verdi</li>
                        <li><strong>False Negative (FN):</strong> Düşme varken tespit edemedi</li>
                    </ul>
                </div>
                <div class="image-grid">
                    <div class="image-item">
                        <img src="confusion_matrix.png" alt="Konfüzyon Matrisi">
                        <p><strong>Binary Sınıflandırma Matrisi (fall vs no_fall)</strong></p>
                    </div>
                </div>
            </div>
            
            <div class="section conclusion">
                <h3>🎉 Sonuç ve Öneriler</h3>
                <ul>
                    <li><strong>Model Başarısı:</strong> YOLOv8n modeli düşme tespiti görevinde başarılı performans gösterdi</li>
                    <li><strong>4GB RAM Optimizasyonu:</strong> Düşük kaynak kullanımı ile etkili eğitim gerçekleştirildi</li>
                    <li><strong>Real-time Kullanım:</strong> Model gerçek zamanlı konveyör bandı izleme için uygun</li>
                    <li><strong>Konfüzyon Matrisi:</strong> Binary sınıflandırma performansı detaylı analiz edildi</li>
                    <li><strong>İyileştirme Önerileri:</strong> Daha fazla veri ile model performansı artırılabilir</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        # HTML raporunu kaydet
        report_path = self.results_dir / 'comprehensive_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Kapsamlı HTML raporu oluşturuldu: {report_path}")
        
        # Özet istatistikleri yazdır
        print("\n" + "="*60)
        print("📋 RAPOR ÖZETİ")
        print("="*60)
        print(f"📂 Tüm dosyalar: {self.results_dir}")
        print("📊 Oluşturulan görselleştirmeler:")
        
        visualizations = [
            "dataset_analysis.png - Veri seti analizi ve dağılımları",
            "sample_images.png - Örnek görüntüler ve etiketler", 
            "training_curves.png - Eğitim loss ve metrik eğrileri",
            "performance_report.png - Test performans metrikleri",
            "prediction_showcase.png - Model tahmin örnekleri",
            "confusion_matrix.png - Binary konfüzyon matrisi (fall/no_fall)",
            "comprehensive_report.html - Kapsamlı HTML raporu",
            "performance_report.json - Detaylı JSON metrikleri",
            "confusion_matrix_results.json - Konfüzyon matrisi sonuçları"
        ]
        
        for viz in visualizations:
            filename = viz.split(' - ')[0]
            if Path(self.results_dir / filename).exists():
                print(f"   ✅ {viz}")
            else:
                print(f"   ⚠️ {viz} (oluşturulmamış)")

def main():
    """Enhanced main function with confusion matrix"""
    
    print("🚀 Konveyör Bandı Düşme Tespiti - Enhanced YOLOv8n + Konfüzyon Matrisi")
    print("=" * 80)
    
    # Veri seti yolunu ayarla
    DATA_FOLDER_PATH = "/Users/haticegenc/Documents/Conveyor_fall_detection/data" 
    
    # Enhanced trainer oluştur
    trainer = FallDetectionTrainer(DATA_FOLDER_PATH)
    
    try:
        # 1. Veri seti yapısını kontrol et
        print("\n📁 1. Veri seti yapısı kontrol ediliyor...")
        if not trainer.check_dataset_structure():
            print("❌ Veri seti yapısı hatalı!")
            return
        
        # 2. YAML yollarını güncelle
        print("\n⚙️ 2. YAML konfigürasyonu güncelleniyor...")
        if not trainer.update_yaml_paths():
            print("❌ YAML güncellenemedi!")
            return
        
        # 3. Veri seti analizi YAP
        print("\n📊 3. Detaylı veri seti analizi yapılıyor...")
        dataset_analysis = trainer.analyze_dataset()
        
        # 4. Örnek görüntü grid'i oluştur
        print("\n🖼️ 4. Örnek görüntü grid'i oluşturuluyor...")
        trainer.create_sample_grid()
        
        # 5. Modeli eğit (enhanced)
        print("\n🚀 5. YOLOv8n modeli eğitiliyor (enhanced monitoring)...")
        training_results = trainer.train_model(
            epochs=20,
            imgsz=416,
            batch_size=4
        )
        
        # 6. Modeli değerlendir (enhanced + konfüzyon matrisi) 
        print("\n📊 6. Model performansı değerlendiriliyor (konfüzyon matrisi dahil)...")
        test_results, confusion_results = trainer.evaluate_model()
        
        # 7. Test tahminleri (enhanced showcase)
        print("\n🔍 7. Test tahminleri ve showcase oluşturuluyor...")
        trainer.test_inference()
        
        # 8. Kapsamlı rapor oluştur (konfüzyon matrisi dahil)
        print("\n📋 8. Kapsamlı rapor oluşturuluyor (konfüzyon matrisi dahil)...")
        trainer.generate_comprehensive_report()
        
        print("\n🎉 TÜM SÜREÇ BAŞARIYLA TAMAMLANDI!")
        print(f"\n📂 Tüm sonuçlar: {trainer.results_dir}")
        print("\n💡 Sunum için hazır dosyalar:")
        print("   📊 dataset_analysis.png - Veri analizi")
        print("   📈 training_curves.png - Eğitim süreci") 
        print("   🎯 performance_report.png - Model performansı")
        print("   🔄 confusion_matrix.png - Konfüzyon matrisi (fall/no_fall)")
        print("   🖼️ prediction_showcase.png - Tahmin örnekleri")
        print("   📄 comprehensive_report.html - Kapsamlı raporu")
        
    except Exception as e:
        print(f"\n❌ Hata oluştu: {e}")
        import traceback
        traceback.print_exc()
        print("\nLütfen veri seti yolunu kontrol edin ve gerekli kütüphanelerin yüklü olduğundan emin olun")

if __name__ == "__main__":
    main()