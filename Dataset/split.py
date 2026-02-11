base_dir = Path("GrapeVineDisease")
output_dir = Path("GrapeVineDataSplits")

# data
temp_dataset = datasets.ImageFolder(base_dir)

# class names
actual_classes = temp_dataset.classes
print(f"Detected classes: {actual_classes}")

# directories with class names
for split in ["train", "val", "test"]:
    for class_name in actual_classes:
        (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)

# files by class
class_files = {cls_name: [] for cls_name in actual_classes}
for img_path, label in temp_dataset.samples:
    class_name = actual_classes[label]
    class_files[class_name].append(img_path)

# split and copy
print("\nSplitting and copying files...")
for class_name, files in class_files.items():
    print(f"\nProcessing {class_name}: {len(files)} images")

    # split: 80% train, 10% val, 10% test
    train_files, temp_files = train_test_split(files, test_size=0.2, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

    # copy files to respective directories
    for img_path in train_files:
        dest = output_dir / "train" / class_name / Path(img_path).name
        shutil.copy2(img_path, dest)

    for img_path in val_files:
        dest = output_dir / "val" / class_name / Path(img_path).name
        shutil.copy2(img_path, dest)

    for img_path in test_files:
        dest = output_dir / "test" / class_name / Path(img_path).name
        shutil.copy2(img_path, dest)

    print(
        f"  Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}"
    )

print("Data splits created successfully!")
print(f"Location: {output_dir}")
