import os
import sys
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import FireSmokeDetector
from models.utils import FireSmokeDataset

class Evaluator:
    def __init__(self, model_path, config_path, data_path='data/fire_smoke'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        
        # Load model
        self.model = FireSmokeDetector(config_path).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Initialize dataset
        self.test_dataset = FireSmokeDataset(
            data_path=data_path,
            img_size=self.config['model']['input_size'],
            augment=False,
            mode='test'
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=FireSmokeDataset.collate_fn
        )
        
        # Class names
        self.class_names = ['fire', 'smoke', 'wall_clean', 'wall_dirty', 'wall_damaged']
        
        # Thresholds
        self.conf_thres = self.config['inference']['conf_thres']
        self.iou_thres = self.config['inference']['iou_thres']
        
    def non_max_suppression(self, predictions):
        """Perform NMS on inference results"""
        output = []
        
        for pred in predictions:
            # Filter by confidence
            conf_mask = pred[:, 4] > self.conf_thres
            pred = pred[conf_mask]
            
            if len(pred) == 0:
                output.append(torch.zeros((0, 6), device=pred.device))
                continue
            
            # Get class predictions
            class_conf, class_pred = pred[:, 5:].max(1, keepdim=True)
            pred = torch.cat((pred[:, :5], class_conf, class_pred.float()), 1)
            
            # Sort by confidence
            conf_sort_index = torch.argsort(pred[:, 4], descending=True)
            pred = pred[conf_sort_index]
            
            # Perform NMS
            boxes = pred[:, :4]
            scores = pred[:, 4]
            
            # Convert to x1y1x2y2
            boxes[:, 0] -= boxes[:, 2] / 2
            boxes[:, 1] -= boxes[:, 3] / 2
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            
            # NMS
            keep = torch.ops.torchvision.nms(boxes, scores, self.iou_thres)
            output.append(pred[keep])
        
        return output
    
    def calculate_metrics(self, all_predictions, all_targets):
        """Calculate evaluation metrics"""
        # Extract predictions and ground truth
        pred_classes = []
        true_classes = []
        
        for preds, targets in zip(all_predictions, all_targets):
            if len(preds) > 0:
                pred_classes.extend(preds[:, 6].cpu().numpy().astype(int))
            
            if len(targets) > 0:
                true_classes.extend(targets[:, 1].cpu().numpy().astype(int))
        
        # Calculate confusion matrix
        if len(pred_classes) > 0 and len(true_classes) > 0:
            cm = confusion_matrix(true_classes, pred_classes, labels=range(len(self.class_names)))
            
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.class_names, 
                       yticklabels=self.class_names)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig('outputs/visualizations/confusion_matrix.png')
            plt.close()
            
            # Classification report
            report = classification_report(true_classes, pred_classes, 
                                         target_names=self.class_names,
                                         output_dict=True)
            
            return report
        else:
            return None
    
    def visualize_predictions(self, image, predictions, targets, save_path):
        """Visualize predictions on image"""
        img = image.copy()
        
        # Draw ground truth boxes (green)
        for target in targets:
            if len(target) > 0:
                x1, y1, x2, y2 = target[2:6].int()
                class_id = int(target[1])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f'GT: {self.class_names[class_id]}', 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw predictions (red)
        for pred in predictions:
            if len(pred) > 0:
                x1, y1, x2, y2 = pred[:4].int()
                conf = pred[4]
                class_id = int(pred[6])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, f'{self.class_names[class_id]}: {conf:.2f}', 
                           (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.imwrite(save_path, img)
    
    def evaluate(self):
        """Run evaluation"""
        all_predictions = []
        all_targets = []
        
        print('Running evaluation...')
        with torch.no_grad():
            for idx, (imgs, targets) in enumerate(tqdm(self.test_loader)):
                imgs = imgs.to(self.device)
                
                # Forward pass
                outputs = self.model(imgs)
                
                # NMS
                predictions = self.non_max_suppression(outputs)
                
                all_predictions.extend(predictions)
                all_targets.append(targets)
                
                # Visualize first few images
                if idx < 10:
                    img = imgs[0].cpu().numpy().transpose(1, 2, 0)
                    img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
                    img = img.astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    
                    save_path = f'outputs/visualizations/test_result_{idx}.jpg'
                    self.visualize_predictions(img, predictions[0], targets, save_path)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_predictions, all_targets)
        
        if metrics:
            print('\nClassification Report:')
            for class_name in self.class_names:
                if class_name in metrics:
                    print(f"{class_name}:")
                    print(f"  Precision: {metrics[class_name]['precision']:.3f}")
                    print(f"  Recall: {metrics[class_name]['recall']:.3f}")
                    print(f"  F1-score: {metrics[class_name]['f1-score']:.3f}")
            
            print(f"\nOverall Accuracy: {metrics['accuracy']:.3f}")
        
        print('\nEvaluation completed!')

def main():
    parser = argparse.ArgumentParser(description='Evaluate Fire and Smoke Detection Model')
    parser.add_argument('--model', type=str, default='outputs/checkpoints/best.pth', help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml', help='Path to config file')
    parser.add_argument('--data', type=str, default='data/fire_smoke', help='Path to test data')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs('outputs/visualizations', exist_ok=True)
    
    # Initialize evaluator
    evaluator = Evaluator(args.model, args.config, args.data)
    
    # Run evaluation
    evaluator.evaluate()

if __name__ == '__main__':
    main()