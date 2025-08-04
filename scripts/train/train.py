import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import wandb

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import FireSmokeDetector
from models.utils import FireSmokeDataset, YOLOLoss

class Trainer:
    def __init__(self, config_path):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device (Mac M-series MPS 지원)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f'Using device: {self.device} (CUDA)')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print(f'Using device: {self.device} (Apple Silicon MPS)')
        else:
            self.device = torch.device('cpu')
            print(f'Using device: {self.device} (CPU)')
        
        print(f'Device: {self.device}')
        
        # Initialize model
        self.model = FireSmokeDetector(config_path).to(self.device)
        
        # Initialize datasets
        self.train_dataset = FireSmokeDataset(
            data_path='data/fire_smoke',
            img_size=self.config['model']['input_size'],
            augment=True,
            mode='train'
        )
        
        self.val_dataset = FireSmokeDataset(
            data_path='data/fire_smoke',
            img_size=self.config['model']['input_size'],
            augment=False,
            mode='val'
        )
        
        # Initialize dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=FireSmokeDataset.collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=FireSmokeDataset.collate_fn
        )
        
        # Initialize loss
        self.criterion = YOLOLoss(
            num_classes=self.config['model']['num_classes'],
            anchors=self.config['model']['anchors'],
            device=self.device,
            box_loss_gain=self.config['training']['loss']['box'],
            cls_loss_gain=self.config['training']['loss']['cls'],
            obj_loss_gain=self.config['training']['loss']['obj']
        )
        
        # Initialize optimizer
        self.optimizer = SGD(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            momentum=self.config['training']['momentum'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Initialize scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['epochs']
        )
        
        # Initialize tensorboard
        self.writer = SummaryWriter('outputs/logs')
        
        # Initialize WandB
        self.use_wandb = self.config.get('wandb', {}).get('enabled', False)
        if self.use_wandb:
            wandb.init(
                project=self.config.get('wandb', {}).get('project', 'FireSmoke'),
                entity=self.config.get('wandb', {}).get('entity', None),
                config=self.config,
                name=f"experiment_{self.config.get('experiment_name', 'default')}",
                tags=['fire-detection', 'yolov5', 'cbam', 'attention']
            )
            # WandB에서 모델 아키텍처 로그
            wandb.watch(self.model, log='all', log_freq=100)
        
        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}/{self.config["training"]["epochs"]}')
        
        for batch_idx, (imgs, targets) in enumerate(pbar):
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(imgs)
            
            # Calculate loss
            loss, loss_items = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'box': f'{loss_items[0]:.4f}',
                'obj': f'{loss_items[1]:.4f}',
                'cls': f'{loss_items[2]:.4f}'
            })
            
            # Log to tensorboard and wandb
            if batch_idx % 10 == 0:
                step = self.epoch * len(self.train_loader) + batch_idx
                
                # TensorBoard
                self.writer.add_scalar('Train/Loss', loss.item(), step)
                self.writer.add_scalar('Train/BoxLoss', loss_items[0], step)
                self.writer.add_scalar('Train/ObjLoss', loss_items[1], step)
                self.writer.add_scalar('Train/ClsLoss', loss_items[2], step)
                
                # WandB
                if self.use_wandb:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/box_loss': loss_items[0],
                        'train/obj_loss': loss_items[1],
                        'train/cls_loss': loss_items[2],
                        'train/step': step,
                        'epoch': self.epoch
                    }, step=step)
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for imgs, targets in pbar:
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(imgs)
                
                # Calculate loss
                loss, loss_items = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}'
                })
        
        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, filename='checkpoint.pth'):
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        torch.save(checkpoint, os.path.join('outputs/checkpoints', filename))
    
    def train(self):
        print(f'Starting training for {self.config["training"]["epochs"]} epochs...')
        print(f'Train samples: {len(self.train_dataset)}, Val samples: {len(self.val_dataset)}')
        
        for epoch in range(self.config['training']['epochs']):
            self.epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # TensorBoard
            self.writer.add_scalar('Epoch/TrainLoss', train_loss, epoch)
            self.writer.add_scalar('Epoch/ValLoss', val_loss, epoch)
            self.writer.add_scalar('Epoch/LearningRate', self.scheduler.get_last_lr()[0], epoch)
            
            # WandB
            if self.use_wandb:
                wandb.log({
                    'epoch/train_loss': train_loss,
                    'epoch/val_loss': val_loss,
                    'epoch/learning_rate': self.scheduler.get_last_lr()[0],
                    'epoch': epoch
                }, step=epoch)
            
            # Save checkpoint
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint('best.pth')
                print(f'New best model saved with loss: {val_loss:.4f}')
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'epoch_{epoch+1}.pth')
        
        # Save final model
        self.save_checkpoint('final.pth')
        
        # WandB 종료
        if self.use_wandb:
            wandb.finish()
        
        print('Training completed!')

def main():
    parser = argparse.ArgumentParser(description='Train Fire and Smoke Detection Model')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs('outputs/checkpoints', exist_ok=True)
    os.makedirs('outputs/logs', exist_ok=True)
    
    # Initialize trainer
    trainer = Trainer(args.config)
    
    # Start training
    trainer.train()

if __name__ == '__main__':
    main()