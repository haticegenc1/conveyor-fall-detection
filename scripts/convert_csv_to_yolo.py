import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm # İlerleme çubuğu için
import shutil # Dosya kopyalama için
import numpy as np # NaN değer kontrolü için

# --- Yapılandırma Ayarları ---
# Script, 'CONVEYOR_FALL_DETECTION/scripts/' klasörünün içinde olduğundan,
# yollar 'CONVEYOR_FALL_DETECTION/' ana dizinine göre göreceli olarak verilmelidir.

# CSV dosyanızın yolu: CONVEYOR_FALL_DETECTION/data/annotations.csv
CSV_PATH = '/Users/haticegenc/Documents/Conveyor_fall_detection/data/annotations.csv'

# Orijinal görüntülerin kök klasörü: CONVEYOR_FALL_DETECTION/data/raw/
# Bu klasör 'fall/' ve 'no_fall/' alt klasörlerini içerir.
IMAGES_DIR = '/Users/haticegenc/Documents/Conveyor_fall_detection/data/raw'

# YOLO formatında işlenmiş verilerin (images/, labels/, data.yaml)
OUTPUT_BASE_DIR = '/Users/haticegenc/Documents/Conveyor_fall_detection/data/processed'

# Sınıf İsimleri ve ID'leri (Kendi sınıflarınıza ve YOLO'daki sıralamaya göre güncelleyin)
# Buradaki sıralama, YOLO'nun .names dosyasındaki sıralama ile aynı olmalıdır.
CLASS_NAMES = ['fall', 'no_fall']

CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}

# Veri Seti Bölme Oranları (Toplamı 1.0 olmalı)
TRAIN_RATIO = 0.7  # Eğitim seti oranı
VAL_RATIO = 0.15   # Doğrulama seti oranı
TEST_RATIO = 0.15  # Test seti oranı

# CSV sütun isimleri (Sizin CSV'nizdeki sütun isimlerine göre güncellendi!)
CSV_COL_IMAGE_NAME = 'image_name'
CSV_COL_CLASS_NAME = 'label' # CSV'nizde bu 'label' olarak geçiyordu
CSV_COL_X_MIN = 'x_min'
CSV_COL_Y_MIN = 'y_min'
CSV_COL_X_MAX = 'x_max'
CSV_COL_Y_MAX = 'y_max'

# --- Fonksiyonlar ---

def convert_bbox_to_yolo(x_min, y_min, x_max, y_max, img_width, img_height):
    """
    Normal piksel koordinatlarını YOLO formatına dönüştürür.
    YOLO formatı: [x_center, y_center, width, height] normalize edilmiş 0-1 aralığında.
    """
    """YOLO formatına dönüştürme fonksiyonu - geliştirilmiş"""
    if any(pd.isna([x_min, y_min, x_max, y_max])):
        return None
    
    # Koordinatların mantıklı olduğunu kontrol et
    if x_max <= x_min or y_max <= y_min:
        return None
        
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / float(img_width)
    height = (y_max - y_min) / float(img_height)
    
    # Koordinatların 0-1 aralığında olduğunu kontrol et
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
        return None
        
    return x_center, y_center, width, height

# --- Fonksiyonlar --- (Bu bölümde)

def convert_bbox_to_yolo(x_min, y_min, x_max, y_max, img_width, img_height):
    """
    Normal piksel koordinatlarını YOLO formatına dönüştürür.
    YOLO formatı: [x_center, y_center, width, height] normalize edilmiş 0-1 aralığında.
    """
    if any(pd.isna([x_min, y_min, x_max, y_max])):
        return None

    # Koordinat mantık kontrolü ekleyin
    if x_max <= x_min or y_max <= y_min:
        return None

    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / float(img_width)
    height = (y_max - y_min) / float(img_height)
    
    # YOLO koordinatlarının geçerli olup olmadığını kontrol et
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
        return None
        
    return x_center, y_center, width, height

# ESKİ create_yolo_dataset FONKSIYONUNU SİLİN VE YERİNE BU KOYUN:
def create_yolo_dataset(csv_path, images_dir, output_base_dir, class_map,
                        train_ratio, val_ratio, test_ratio,
                        csv_col_img_name, csv_col_class_name,
                        csv_col_x_min, csv_col_y_min, csv_col_x_max, csv_col_y_max):

    print(f"'{csv_path}' adresindeki CSV dosyasını okuyor...")
    df = pd.read_csv(csv_path)
    
    # İstatistik değişkenleri
    total_bboxes = len(df)
    processed_bboxes = 0
    skipped_bboxes = 0
    images_with_no_valid_bboxes = 0
    
    print(f"Toplam bounding box: {total_bboxes}")

    image_groups = df.groupby(csv_col_img_name)

    # Çıktı klasörlerini oluştur
    for subset in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_base_dir, 'images', subset), exist_ok=True)
        os.makedirs(os.path.join(output_base_dir, 'labels', subset), exist_ok=True)

    all_image_names = list(image_groups.groups.keys())
    
    train_names, temp_names = train_test_split(all_image_names, test_size=(val_ratio + test_ratio), random_state=42)
    val_test_ratio_split = test_ratio / (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 0
    val_names, test_names = train_test_split(temp_names, test_size=val_test_ratio_split, random_state=42)

    train_names_set = set(train_names)
    val_names_set = set(val_names)
    test_names_set = set(test_names)

    print("Görüntüleri ve etiketleri işliyor ve kopyalıyor...")
    
    for img_name, group_df in tqdm(image_groups, desc="Görüntüler işleniyor"):
        original_class_folder_name = group_df[csv_col_class_name].iloc[0] 
        img_path = os.path.join(images_dir, original_class_folder_name, img_name)
        
        # Görüntünün hangi sete ait olduğunu belirle
        current_subset = None
        if img_name in train_names_set:
            current_subset = 'train'
        elif img_name in val_names_set:
            current_subset = 'val'
        elif img_name in test_names_set:
            current_subset = 'test'
        else:
            print(f"Uyarı: '{img_name}' hiçbir sete atanamadı, atlanıyor.")
            continue

        output_img_dir = os.path.join(output_base_dir, 'images', current_subset)
        output_label_dir = os.path.join(output_base_dir, 'labels', current_subset)
        
        try:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
        except FileNotFoundError:
            print(f"Uyarı: '{img_path}' görüntüsü bulunamadı, atlanıyor.")
            continue
        except Exception as e:
            print(f"Uyarı: '{img_path}' görüntüsü açılırken bir hata oluştu: {e}, atlanıyor.")
            continue

        label_file_name = os.path.splitext(img_name)[0] + '.txt'
        label_file_path = os.path.join(output_label_dir, label_file_name)

        yolo_lines = []
        valid_bboxes_for_this_image = 0
        
        for idx, row in group_df.iterrows():
            obj_class_name = row[csv_col_class_name]
            class_id = class_map.get(obj_class_name)
            
            if class_id is None:
                print(f"Uyarı: Bilinmeyen sınıf '{obj_class_name}' bulundu, '{img_name}' görüntüsündeki bu etiket atlanıyor.")
                skipped_bboxes += 1
                continue

            x_min, y_min, x_max, y_max = (
                row[csv_col_x_min], row[csv_col_y_min],
                row[csv_col_x_max], row[csv_col_y_max]
            )

            # NaN kontrolü
            if pd.isna(x_min) or pd.isna(y_min) or pd.isna(x_max) or pd.isna(y_max):
                skipped_bboxes += 1
                continue
            
            # Koordinat mantık kontrolü
            if x_max <= x_min or y_max <= y_min:
                print(f"Uyarı: '{img_name}' - Geçersiz koordinatlar: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
                skipped_bboxes += 1
                continue
            
            # Koordinatların görüntü sınırları içinde olup olmadığını kontrol et
            if x_min < 0 or y_min < 0 or x_max > img_width or y_max > img_height:
                print(f"Uyarı: '{img_name}' - Koordinatlar görüntü sınırları dışında: ({x_min},{y_min},{x_max},{y_max}), görüntü boyutu: {img_width}x{img_height}")
                skipped_bboxes += 1
                continue

            yolo_coords = convert_bbox_to_yolo(
                x_min, y_min, x_max, y_max, img_width, img_height
            )

            if yolo_coords is None:
                print(f"Uyarı: '{img_name}' görüntüsündeki bounding box {x_min},{y_min},{x_max},{y_max} için geçersiz YOLO koordinatları döndü, atlanıyor.")
                skipped_bboxes += 1
                continue
            
            x_center, y_center, width, height = yolo_coords
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            processed_bboxes += 1
            valid_bboxes_for_this_image += 1
        
        # Eğer bu görüntü için hiç geçerli bounding box yoksa
        if valid_bboxes_for_this_image == 0:
            images_with_no_valid_bboxes += 1
            print(f"Uyarı: '{img_name}' için hiç geçerli bounding box bulunamadı!")
        
        # Etiket dosyasını yaz
        with open(label_file_path, 'w') as f:
            f.writelines(yolo_lines)
        
        # Görüntüyü kopyala
        shutil.copy(img_path, os.path.join(output_img_dir, img_name))
            
    print("\nYOLO etiketleme ve veri seti bölme tamamlandı!")

    # Detaylı istatistikler
    print(f"\n=== DETAYLI İSTATİSTİKLER ===")
    print(f"Toplam bounding box: {total_bboxes}")
    print(f"İşlenen (geçerli) bounding box: {processed_bboxes}")
    print(f"Atlanan (geçersiz) bounding box: {skipped_bboxes}")
    print(f"Geçerli bounding box oranı: %{(processed_bboxes/total_bboxes)*100:.2f}")
    print(f"Hiç geçerli bounding box'ı olmayan görüntü sayısı: {images_with_no_valid_bboxes}")
    
    print(f"\n=== GÖRÜNTÜ DAĞILIMI ===")
    print(f"Toplam görüntü: {len(all_image_names)}")
    print(f"Eğitim seti: {len(train_names)} (%{len(train_names)/len(all_image_names)*100:.1f})")
    print(f"Doğrulama seti: {len(val_names)} (%{len(val_names)/len(all_image_names)*100:.1f})")
    print(f"Test seti: {len(test_names)} (%{len(test_names)/len(all_image_names)*100:.1f})")

    # data.yaml oluştur
    data_yaml_path = os.path.join(output_base_dir, 'data.yaml')
    with open(data_yaml_path, 'w') as f:
        f.write(f"path: {os.path.abspath(output_base_dir)}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"test: images/test\n\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")

    print(f"'{data_yaml_path}' dosyası oluşturuldu.")
    print("Artık YOLO modelinizi eğitmeye hazırsınız!")

# --- Script'i Çalıştır --- (Bu bölüm aynı kalacak)

# --- Script'i Çalıştır ---
if __name__ == "__main__":
    create_yolo_dataset(
        csv_path=CSV_PATH,
        images_dir=IMAGES_DIR,
        output_base_dir=OUTPUT_BASE_DIR,
        class_map=CLASS_MAP,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        csv_col_img_name=CSV_COL_IMAGE_NAME,
        csv_col_class_name=CSV_COL_CLASS_NAME,
        csv_col_x_min=CSV_COL_X_MIN,
        csv_col_y_min=CSV_COL_Y_MIN,
        csv_col_x_max=CSV_COL_X_MAX,
        csv_col_y_max=CSV_COL_Y_MAX
    )
    
