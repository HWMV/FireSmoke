import torch
import torch.nn as nn
import numpy as np
from .attention_modules import CBAM, SEBlock, ECA

class AttentionHead(nn.Module):
    def __init__(self, in_channels, num_classes, anchors, attention_type='CBAM', reduction_ratio=16):
        super(AttentionHead, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = len(anchors[0]) // 2
        self.anchors = anchors
        self.stride = [8, 16, 32]  # P3, P4, P5
        
        # Attention modules for each scale
        self.attention_modules = nn.ModuleList()
        for ch in in_channels:
            if attention_type == 'CBAM':
                self.attention_modules.append(CBAM(ch, reduction_ratio))
            elif attention_type == 'SE':
                self.attention_modules.append(SEBlock(ch, reduction_ratio))
            elif attention_type == 'ECA':
                self.attention_modules.append(ECA(ch))
            else:
                raise ValueError(f"Unknown attention type: {attention_type}")
        
        # Detection heads for each scale
        self.detection_heads = nn.ModuleList()
        for i, ch in enumerate(in_channels):
            self.detection_heads.append(
                nn.Conv2d(ch, self.num_anchors * (5 + num_classes), 1)
            )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, features):
        outputs = []
        
        for i, (feature, attention, head) in enumerate(zip(features, self.attention_modules, self.detection_heads)):
            # Apply attention
            feature = attention(feature)
            
            # Apply detection head
            output = head(feature)
            
            # Reshape output
            bs, _, h, w = output.shape
            output = output.view(bs, self.num_anchors, 5 + self.num_classes, h, w)
            output = output.permute(0, 1, 3, 4, 2).contiguous()
            
            # Apply activations
            if not self.training:
                # During inference, apply sigmoid to objectness and class predictions
                output[..., 4:] = torch.sigmoid(output[..., 4:])
                
                # Convert box predictions to absolute coordinates
                grid = self._make_grid(w, h, feature.dtype, feature.device)
                anchor_grid = torch.tensor(self.anchors[i], dtype=feature.dtype, device=feature.device).view(1, self.num_anchors, 1, 1, 2)
                
                xy = (output[..., :2] * 2 - 0.5 + grid) * self.stride[i]
                wh = (output[..., 2:4] * 2) ** 2 * anchor_grid
                output = torch.cat((xy, wh, output[..., 4:]), -1)
            
            outputs.append(output.view(bs, -1, 5 + self.num_classes))
        
        return torch.cat(outputs, 1) if not self.training else outputs
    
    @staticmethod
    def _make_grid(w, h, dtype, device):
        yv, xv = torch.meshgrid([torch.arange(h, device=device), torch.arange(w, device=device)], indexing='ij')
        return torch.stack((xv, yv), 2).view(1, 1, h, w, 2).to(dtype)