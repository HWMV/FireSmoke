import torch
import torch.nn as nn
import yaml
from .backbone import YOLOv5Backbone
from .heads import AttentionHead

class FireSmokeDetector(nn.Module):
    def __init__(self, config_path=None):
        super(FireSmokeDetector, self).__init__()
        
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration
            self.config = {
                'model': {
                    'num_classes': 5,
                    'width_multiple': 0.25,
                    'depth_multiple': 0.33,
                    'head': {
                        'type': 'AttentionHead',
                        'attention_type': 'CBAM',
                        'reduction_ratio': 16
                    },
                    'anchors': [
                        [10,13, 16,30, 33,23],
                        [30,61, 62,45, 59,119],
                        [116,90, 156,198, 373,326]
                    ]
                }
            }
        
        # Build model
        self.backbone = YOLOv5Backbone(
            width_multiple=self.config['model']['width_multiple'],
            depth_multiple=self.config['model']['depth_multiple']
        )
        
        self.head = AttentionHead(
            in_channels=self.backbone.out_channels,
            num_classes=self.config['model']['num_classes'],
            anchors=self.config['model']['anchors'],
            attention_type=self.config['model']['head']['attention_type'],
            reduction_ratio=self.config['model']['head']['reduction_ratio']
        )
        
    def forward(self, x):
        features = self.backbone(x)
        outputs = self.head(features)
        return outputs
    
    def fuse(self):
        print('Fusing layers...')
        for m in self.modules():
            if type(m) is nn.Conv2d and hasattr(m, 'bn'):
                m.conv = self._fuse_conv_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.forward_fuse
        return self
    
    @staticmethod
    def _fuse_conv_bn(conv, bn):
        fusedconv = nn.Conv2d(conv.in_channels,
                             conv.out_channels,
                             kernel_size=conv.kernel_size,
                             stride=conv.stride,
                             padding=conv.padding,
                             groups=conv.groups,
                             bias=True).requires_grad_(False).to(conv.weight.device)
        
        # Prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
        
        # Prepare bias
        b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
        
        return fusedconv