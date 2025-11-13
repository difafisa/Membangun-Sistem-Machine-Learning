import os
import shutil
from sklearn.model_selection import train_test_split

def sync_image_label(image_dir, label_dir):
    print("Menyamakan file image dan label...")
    images = set(os.path.splitext(f)[0] for f in os.listdir(image_dir))
    labels = set(os.path.splitext(f)[0] for f in os.listdir(label_dir))
    common = images & labels
    print(f"Ditemukan {len(common)} pasangan image-label yang cocok.\n")
    return list(common)

def split_dataset(file_list, test_size=0.3, val_size=0.2, seed=42):
    print("Membagi dataset menjadi train, val, dan test...")
    train_imgs, temp_imgs = train_test_split(file_list, test_size=test_size, random_state=seed)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=val_size/(test_size), random_state=seed)
    print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}\n")
    return train_imgs, val_imgs, test_imgs

def copy_files(file_list, src_img, src_lbl, dst_img, dst_lbl):
    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lbl, exist_ok=True)
    for f in file_list:
        shutil.copy(f"{src_img}/{f}.jpg", f"{dst_img}/{f}.jpg")
        shutil.copy(f"{src_lbl}/{f}.txt", f"{dst_lbl}/{f}.txt")

def main():
    print("Mulai proses preprocessing dataset...\n")

    src_img = '../vehicles_raw/train/images'
    src_lbl = '../vehicles_raw/train/labels'
    file_list = sync_image_label(src_img, src_lbl)

    train_imgs, val_imgs, test_imgs = split_dataset(file_list)

    print("Menyalin file ke folder output...")
    copy_files(train_imgs, src_img, src_lbl, 'dataset_preprocessing/train/images', 'dataset_preprocessing/train/labels')
    copy_files(val_imgs, src_img, src_lbl, 'dataset_preprocessing/val/images', 'dataset_preprocessing/val/labels')
    copy_files(test_imgs, src_img, src_lbl, 'dataset_preprocessing/test/images', 'dataset_preprocessing/test/labels')

    print("\nPreprocessing selesai! Dataset siap digunakan untuk pelatihan model.")

if __name__ == "__main__":
    main()
