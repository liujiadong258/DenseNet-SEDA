import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.cuda.amp import autocast
from dataset import MineralDataset
from model import DenseNet_SEDA
torch.set_num_threads(8)

#验证与测试主函数
@torch.no_grad()
def evaluate():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(base_dir, "Photo", "test")
    ckpt_path = os.path.join(base_dir, "checkpoints", "best_model.pth")

    if not os.path.exists(ckpt_path):
        print(f"Error: 权重文件不存在: {ckpt_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = MineralDataset(test_dir, transform=test_tf)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    model = DenseNet_SEDA(num_classes=len(test_dataset.classes), pretrained=False).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(test_loader, desc="Testing", bar_format="{l_bar}{bar:30}{r_bar}")
    start_time = time.time()

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "Loss": f"{total_loss / total:.4f}",
            "Acc": f"{correct / total * 100:.2f}%"
        })

    elapsed = time.time() - start_time
    test_acc = correct / total
    test_loss = total_loss / total

    print("-" * 50)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.2f}% | Time: {elapsed:.1f}s")
    print("-" * 50)

if __name__ == "__main__":
    evaluate()