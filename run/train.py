import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from configs.config import Config
from networks.DeepLabFactored import DeepLabFactored
from utils.losses import crossentropy_loss
from src.train.scheduler import LR_Scheduler
from src.dataset import SegmentationDataset
from src.utils import AverageMeter, Evaluator

def train_epoch(model, dataloader, optimizer, criterion, scheduler, device, epoch):
    model.train()
    losses = AverageMeter()
    evaluator = Evaluator(num_class=Config.num_classes)  # Set based on number of classes
    for i, sample in enumerate(dataloader):
        images = sample['image'].to(device)
        labels = sample['anim'].to(device)  # Assuming 'anim' label is the target

        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), images.size(0))
        evaluator.add_batch(labels.cpu().numpy(), outputs.argmax(dim=1).cpu().numpy())
        
        scheduler(optimizer, i, epoch)

    print(f"Epoch {epoch}: Loss={losses.avg:.4f}, mIoU={evaluator.Mean_Intersection_over_Union():.4f}")

def main():
    # Load config
    config = Config()
    device = config.device

    # Initialize model, criterion, optimizer, scheduler
    model = DeepLabFactored(num_anim_classes=config.num_anim_classes, num_inanim_classes=config.num_inanim_classes,
                            output_stride=config.output_stride).to(device)
    criterion = crossentropy_loss
    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = LR_Scheduler(mode=config.lr_scheduler_mode, base_lr=config.lr, num_epochs=config.num_epochs, 
                             iters_per_epoch=config.iters_per_epoch, lr_step=config.lr_step)

    # Initialize DataLoader
    train_dataset = SegmentationDataset(config.data_folder, mode='train', 
                                        input_shape=config.input_shape, num_classes=config.num_classes)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)

    # Training Loop
    for epoch in range(config.num_epochs):
        train_epoch(model, train_loader, optimizer, criterion, scheduler, device, epoch)

    # Save model
    torch.save(model.state_dict(), config.model_save_path)
    print(f"Training complete. Model saved to {config.model_save_path}")

if __name__ == "__main__":
    main()
