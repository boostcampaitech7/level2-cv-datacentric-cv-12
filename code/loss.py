import torch
import torch.nn as nn


def get_dice_loss(gt_score, pred_score):
    """
    Dice 손실을 계산하는 함수입니다.
    
    입력:
        gt_score (torch.Tensor): Ground Truth 스코어 맵
        pred_score (torch.Tensor): 예측된 스코어 맵
    
    출력:
        torch.Tensor: Dice 손실 값
    """
    inter = torch.sum(gt_score * pred_score)  # 교집합 계산
    union = torch.sum(gt_score) + torch.sum(pred_score) + 1e-5  # 합집합 계산 (안전한 분모를 위해 작은 값 추가)
    return 1. - (2 * inter / union)  # Dice 손실 반환


def get_geo_loss(gt_geo, pred_geo):
    """
    지오메트리 손실을 계산하는 함수입니다. IoU 손실과 각도 손실을 포함합니다.
    
    입력:
        gt_geo (torch.Tensor): Ground Truth 지오메트리 맵 (d1, d2, d3, d4, angle)
        pred_geo (torch.Tensor): 예측된 지오메트리 맵 (d1, d2, d3, d4, angle)
    
    출력:
        tuple: IoU 손실 맵과 각도 손실 맵
    """
    # Ground Truth와 예측 지오메트리 맵을 분할
    d1_gt, d2_gt, d3_gt, d4_gt, angle_gt = torch.split(gt_geo, 1, 1)
    d1_pred, d2_pred, d3_pred, d4_pred, angle_pred = torch.split(pred_geo, 1, 1)
    
    # 면적 계산
    area_gt = (d1_gt + d2_gt) * (d3_gt + d4_gt)
    area_pred = (d1_pred + d2_pred) * (d3_pred + d4_pred)
    
    # 교집합 면적 계산
    w_union = torch.min(d3_gt, d3_pred) + torch.min(d4_gt, d4_pred)
    h_union = torch.min(d1_gt, d1_pred) + torch.min(d2_gt, d2_pred)
    area_intersect = w_union * h_union
    
    # 합집합 면적 계산
    area_union = area_gt + area_pred - area_intersect
    
    # IoU 손실 계산
    iou_loss_map = -torch.log((area_intersect + 1.0) / (area_union + 1.0))
    
    # 각도 손실 계산
    angle_loss_map = 1 - torch.cos(angle_pred - angle_gt)
    
    return iou_loss_map, angle_loss_map


class EASTLoss(nn.Module):
    """
    EAST 모델을 위한 손실 함수 클래스입니다.
    
    이 클래스는 분류 손실(Dice Loss)과 지오메트리 손실(IoU Loss 및 각도 손실)을 결합하여 총 손실을 계산합니다.
    """
    def __init__(self, weight_angle=10):
        """
        EASTLoss 클래스의 초기화 함수입니다.
        
        입력:
            weight_angle (float, optional): 각도 손실에 적용할 가중치 (기본값: 10)
        """
        super().__init__()
        self.weight_angle = weight_angle

    def forward(self, gt_score, pred_score, gt_geo, pred_geo, roi_mask):
        """
        손실을 계산하는 순전파 함수입니다.
        
        입력:
            gt_score (torch.Tensor): Ground Truth 스코어 맵
            pred_score (torch.Tensor): 예측된 스코어 맵
            gt_geo (torch.Tensor): Ground Truth 지오메트리 맵
            pred_geo (torch.Tensor): 예측된 지오메트리 맵
            roi_mask (torch.Tensor): ROI 마스크
            
        출력:
            tuple:
                total_loss (torch.Tensor): 총 손실 값
                loss_dict (dict): 개별 손실 값 (분류 손실, 각도 손실, IoU 손실)
        """
        # Ground Truth 스코어 맵에 유효한 점이 없는 경우 손실을 0으로 설정
        if torch.sum(gt_score) < 1:
            return torch.sum(pred_score + pred_geo) * 0, dict(cls_loss=None, angle_loss=None, iou_loss=None)
        
        # 분류 손실(Dice Loss) 계산
        classify_loss = get_dice_loss(gt_score, pred_score * roi_mask)
        
        # 지오메트리 손실(IoU Loss 및 각도 손실) 계산
        iou_loss_map, angle_loss_map = get_geo_loss(gt_geo, pred_geo)
        
        # 각도 손실 계산 및 가중치 적용
        angle_loss = torch.sum(angle_loss_map * gt_score) / torch.sum(gt_score)
        angle_loss *= self.weight_angle
        
        # IoU 손실 계산
        iou_loss = torch.sum(iou_loss_map * gt_score) / torch.sum(gt_score)
        
        # 총 지오메트리 손실 계산
        geo_loss = angle_loss + iou_loss
        
        # 총 손실 계산 (분류 손실 + 지오메트리 손실)
        total_loss = classify_loss + geo_loss
    
        # 손실 값과 개별 손실을 딕셔너리로 반환
        return total_loss, dict(cls_loss=classify_loss.item(), angle_loss=angle_loss.item(),
                                iou_loss=iou_loss.item())
