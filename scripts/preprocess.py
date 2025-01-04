import os
import shutil
from sklearn.model_selection import train_test_split

def prepare_data(input_dir, output_dir, split_ratios=(0.7, 0.15, 0.15)):
    categories = ["TB", "Normal"]
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")

    for category in categories:
        category_path = os.path.join(input_dir, category)
        images = os.listdir(category_path)
        train, temp = train_test_split(images, test_size=split_ratios[1] + split_ratios[2])
        val, test = train_test_split(temp, test_size=split_ratios[2] / (split_ratios[1] + split_ratios[2]))

        for subset, subset_dir in zip([train, val, test], [train_dir, val_dir, test_dir]):
            os.makedirs(os.path.join(subset_dir, category), exist_ok=True)
            for img in subset:
                shutil.copy(os.path.join(category_path, img), os.path.join(subset_dir, category, img))

if __name__ == "__main__":
    prepare_data("data_raw", "data")
