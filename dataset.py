import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

#矿石基础数据集类
class MineralDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = os.path.abspath(root_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []

        self.classes = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d)) and d.startswith("class_")
        ])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')

        for cls_name in self.classes:
            cls_folder = os.path.join(self.root_dir, cls_name)
            for fname in sorted(os.listdir(cls_folder)):
                if fname.lower().endswith(valid_exts):
                    self.image_paths.append(os.path.join(cls_folder, fname))
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

#数据集划分及变换子类
class SubDataset(Dataset):
    def __init__(self, base_dataset, indices, transform=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        path = self.base_dataset.image_paths[real_idx]
        image = Image.open(path).convert('RGB')
        label = self.base_dataset.labels[real_idx]
        if self.transform:
            image = self.transform(image)
        return image, label

#数据预处理
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_test_transform

#构建数据加载器
def build_dataloaders(photo_root=None, batch_size=16, seed=42):
    if photo_root is None:
        photo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Photo"))

    train_dir = os.path.join(photo_root, "train")
    test_dir = os.path.join(photo_root, "test")
    train_tf, val_test_tf = get_transforms()

    full_train = MineralDataset(train_dir, transform=None)
    total_len = len(full_train)
    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(total_len, generator=generator).tolist()

    split = int(0.8 * total_len)
    train_indices = shuffled_indices[:split]
    val_indices = shuffled_indices[split:]

    train_set = SubDataset(full_train, train_indices, transform=train_tf)
    val_set = SubDataset(full_train, val_indices, transform=val_test_tf)
    test_set = MineralDataset(test_dir, transform=val_test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, full_train.classes

if __name__ == "__main__":
    t_ldr, v_ldr, te_ldr, clss = build_dataloaders(batch_size=16)
    print(f"Classes: {len(clss)}, Train: {len(t_ldr.dataset)}, Val: {len(v_ldr.dataset)}, Test: {len(te_ldr.dataset)}")