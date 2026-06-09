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

# class mapping
class_mapping = {"Fresh": 0, "Stage1": 1, "Stage-2": 2, "Stage-3": 3}

# train data transformations and augmentation
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  # gorizontal flip
        transforms.RandomVerticalFlip(),  # vertical flip
        transforms.RandomRotation(20),  # random rotation ±20°
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),  # color changes
        transforms.RandomAffine(
            degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5
        ),  # slight affine transforms
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# validation and test transformations , no augmentation
val_test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# load datasets
train_dataset = datasets.ImageFolder(
    "/content/drive/MyDrive/TermPaper/GrapeVineDataSplits/train",
    transform=train_transform,
)
val_dataset = datasets.ImageFolder(
    "/content/drive/MyDrive/TermPaper/GrapeVineDataSplits/val",
    transform=val_test_transform,
)
test_dataset = datasets.ImageFolder(
    "/content/drive/MyDrive/TermPaper/GrapeVineDataSplits/test",
    transform=val_test_transform,
)

# updating labels according to class mapping
for dataset in [train_dataset, val_dataset, test_dataset]:
    dataset.samples = [
        (path, class_mapping[Path(path).parent.name]) for path, _ in dataset.samples
    ]
    dataset.targets = [s[1] for s in dataset.samples]
    dataset.class_to_idx = class_mapping
    dataset.classes = list(class_mapping.keys())

# data loaders
train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)

print(
    f"Final counts:\ntraining: {len(train_dataset)} | validation: {len(val_dataset)} | testing: {len(test_dataset)}"
)
print(f"class mapping: {train_dataset.class_to_idx}")
print(f"classes: {train_dataset.classes}")
