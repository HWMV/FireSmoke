import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOLoss(nn.Module):
    def __init__(self, num_classes, anchors, device, box_loss_gain=0.05, cls_loss_gain=0.5, obj_loss_gain=1.0):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.device = device
        self.box_loss_gain = box_loss_gain
        self.cls_loss_gain = cls_loss_gain
        self.obj_loss_gain = obj_loss_gain
        
        # Loss functions
        self.BCEcls = nn.BCEWithLogitsLoss(reduction='none')
        self.BCEobj = nn.BCEWithLogitsLoss(reduction='none')
        
        # Anchor settings
        self.num_anchors = 3
        self.num_layers = 3
        self.stride = [8, 16, 32]
        
    def forward(self, predictions, targets):
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        
        # Build targets
        tcls, tbox, indices, anchors = self.build_targets(predictions, targets)
        
        # Losses
        for i, pred in enumerate(predictions):  # layer index
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pred[..., 0], device=device)  # target obj
            
            n = b.shape[0]  # number of targets
            if n:
                ps = pred[b, a, gj, gi]  # prediction subset corresponding to targets
                
                # Box regression
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = self.bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss
                
                # Objectness
                tobj[b, a, gj, gi] = iou.detach().clamp(0).type(tobj.dtype)  # iou as objectness target
                
                # Classification
                if self.num_classes > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], 0.0, device=device)  # targets
                    t[range(n), tcls[i]] = 1.0
                    lcls += self.BCEcls(ps[:, 5:], t).mean()  # BCE
            
            obji = self.BCEobj(pred[..., 4], tobj)
            lobj += obji.mean()  # obj loss
        
        # Loss multipliers
        lbox *= self.box_loss_gain
        lobj *= self.obj_loss_gain
        lcls *= self.cls_loss_gain
        
        loss = lbox + lobj + lcls
        return loss, torch.cat((lbox, lobj, lcls, loss)).detach()
    
    def build_targets(self, predictions, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        
        for i in range(self.num_layers):
            anchors = torch.tensor(self.anchors[i], device=targets.device).float().view(self.num_anchors, 2)
            
            # Get grid size
            _, _, h, w = predictions[i].shape
            gain[2:6] = torch.tensor([w, h, w, h])
            
            # Scale targets
            t = targets * gain
            
            if t.shape[0]:
                # Match targets to anchors
                r = t[:, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < 4  # compare aspect ratio
                t = t[j]  # filter
                
                # Get offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < 0.5) & (gxy > 1)).T
                l, m = ((gxi % 1 < 0.5) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float() * 0.5)[j]
            else:
                t = targets[0]
                offsets = 0
            
            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices
            
            # Append
            indices.append((b, torch.ones_like(b).long(), gj.clamp_(0, h - 1), gi.clamp_(0, w - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[0])
            tcls.append(c)  # class
        
        return tcls, tbox, indices, anch
    
    @staticmethod
    def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
        # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
        box2 = box2.T
        
        # Get the coordinates of bounding boxes
        if x1y1x2y2:  # x1, y1, x2, y2 = box1
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        else:  # transform from xywh to xyxy
            b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
            b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
            b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
            b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
        
        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
        
        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps
        
        iou = inter / union
        
        if CIoU or DIoU or GIoU:
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
            if CIoU or DIoU:  # Distance or Complete IoU
                c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                        (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
                if CIoU:
                    v = (4 / torch.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                    with torch.no_grad():
                        alpha = v / (v - iou + (1 + eps))
                    return iou - (rho2 / c2 + v * alpha)  # CIoU
                return iou - rho2 / c2  # DIoU
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        return iou  # IoU