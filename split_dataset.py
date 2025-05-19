import os
import random
import shutil
from pathlib import Path

def split_dataset(image_dir, label_dir, output_img_dir, output_lbl_dir, val_ratio=0.2, seed=42):
    random.seed(seed)

    # Tüm image dosyalarını oku
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    total = len(image_files)
    val_count = int(total * val_ratio)

    # Karıştır ve ayır
    random.shuffle(image_files)
    val_images = image_files[:val_count]
    train_images = image_files[val_count:]

    def copy_files(file_list, split_type):
        for img_file in file_list:
            lbl_file = img_file.rsplit('.', 1)[0] + ".txt"

            src_img = os.path.join(image_dir, img_file)
            src_lbl = os.path.join(label_dir, lbl_file)

            dst_img = os.path.join(output_img_dir, split_type, img_file)
            dst_lbl = os.path.join(output_lbl_dir, split_type, lbl_file)

            # Klasör oluştur
            os.makedirs(os.path.dirname(dst_img), exist_ok=True)
            os.makedirs(os.path.dirname(dst_lbl), exist_ok=True)

            shutil.copy2(src_img, dst_img)
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, dst_lbl)
            else:
                print(f"[UYARI] Label bulunamadı: {lbl_file}")

    # Kopyalama işlemleri
    copy_files(train_images, "train")
    copy_files(val_images, "val")

    print(f"\n✅ Toplam {total} dosya bulundu.")
    print(f"📦 Eğitim için: {len(train_images)}")
    print(f"🧪 Doğrulama için: {len(val_images)}")
    print("Klasörler başarıyla dolduruldu.")

if __name__ == "__main__":
    split_dataset(
        image_dir="dataset/images/all",
        label_dir="dataset/labels/all",
        output_img_dir="images",
        output_lbl_dir="labels"
    )
