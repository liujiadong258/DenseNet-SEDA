# train.py
import os
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from dataset import build_dataloaders
from model import DenseNet_SEDA
torch.set_num_threads(8)

#训练单轮
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(dataloader, desc="Train", bar_format="{l_bar}{bar:30}{r_bar}", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            out_main, out_aux = model(images)
            loss_main = criterion(out_main, labels)
            loss_aux = criterion(out_aux, labels)
            loss = loss_main + 0.4 * loss_aux

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(out_main, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"Loss": f"{total_loss / total:.4f}", "Acc": f"{correct / total * 100:.2f}%"})

    return total_loss / total, correct / total

#验证集评估
@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(dataloader, desc="Val  ", bar_format="{l_bar}{bar:30}{r_bar}", leave=False)
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

        pbar.set_postfix({"Loss": f"{total_loss / total:.4f}", "Acc": f"{correct / total * 100:.2f}%"})

    return total_loss / total, correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    photo_dir = os.path.join(base_dir, "Photo")
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    log_dir = os.path.join(base_dir, "logs")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    batch_size = 16
    epochs = 80
    lr = 1e-4
    weight_decay = 1e-4

    train_loader, val_loader, _, classes = build_dataloaders(photo_root=photo_dir, batch_size=batch_size)

    model = DenseNet_SEDA(num_classes=len(classes), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler()
    writer = SummaryWriter(log_dir=log_dir)

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        cur_lr = scheduler.get_last_lr()[0]

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start_time

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Acc/Train", train_acc, epoch)
        writer.add_scalar("Acc/Val", val_acc, epoch)
        writer.add_scalar("LR", cur_lr, epoch)

        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'current_loss': val_loss,
            'val_acc': val_acc
        }

        torch.save(state, os.path.join(ckpt_dir, "latest_epoch_model.pth"))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(state, os.path.join(ckpt_dir, "best_model.pth"))

        print(f"Epoch [{epoch:02d}/{epochs}] ({elapsed:.1f}s) | LR: {cur_lr:.6f} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc * 100:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc * 100:.2f}% (Best: {best_acc * 100:.2f}%)")

    writer.close()

if __name__ == "__main__":
    main()