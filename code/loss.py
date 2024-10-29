import torch
import torch.nn as nn


def get_dice_loss(gt_score, pred_score):
    '''
    Dice 손실을 계산합니다.
    
    Args:
        gt_score (torch.Tensor): 실제 스코어 맵 (Ground Truth).
        pred_score (torch.Tensor): 예측된 스코어 맵.
    
    Returns:
        torch.Tensor: Dice 손실 값.
    '''
    inter = torch.sum(gt_score * pred_score)  # 교집합 계산
    union = torch.sum(gt_score) + torch.sum(pred_score) + 1e-5  # 합집합 계산 (안정성을 위해 작은 값 추가)
    return 1. - (2 * inter / union)  # Dice 손실 반환


def get_geo_loss(gt_geo, pred_geo):
    '''
    지오메트리 손실을 계산합니다.
    
    Args:
        gt_geo (torch.Tensor): 실제 지오메트리 맵 (Ground Truth).
        pred_geo (torch.Tensor): 예측된 지오메트리 맵.
    
    Returns:
        tuple: IoU 손실 맵과 각도 손실 맵.
    '''
    # 실제 지오메트리 맵과 예측된 지오메트리 맵을 각각 분할
    d1_gt, d2_gt, d3_gt, d4_gt, angle_gt = torch.split(gt_geo, 1, 1)
    d1_pred, d2_pred, d3_pred, d4_pred, angle_pred = torch.split(pred_geo, 1, 1)
    
    # 실제 영역과 예측 영역의 면적 계산
    area_gt = (d1_gt + d2_gt) * (d3_gt + d4_gt)
    area_pred = (d1_pred + d2_pred) * (d3_pred + d4_pred)
    
    # 교집합 영역의 너비와 높이 계산
    w_union = torch.min(d3_gt, d3_pred) + torch.min(d4_gt, d4_pred)
    h_union = torch.min(d1_gt, d1_pred) + torch.min(d2_gt, d2_pred)
    
    # 교집합 영역의 면적 계산
    area_intersect = w_union * h_union
    # 합집합 영역의 면적 계산
    area_union = area_gt + area_pred - area_intersect
    
    # IoU 손실 맵 계산
    iou_loss_map = -torch.log((area_intersect + 1.0) / (area_union + 1.0))
    
    # 각도 손실 맵 계산 (각도의 차이에 따른 손실)
    angle_loss_map = 1 - torch.cos(angle_pred - angle_gt)
    
    return iou_loss_map, angle_loss_map


class EASTLoss(nn.Module):
    '''
    EAST 모델을 위한 커스텀 손실 함수 클래스.
    
    Attributes:
        weight_angle (float): 각도 손실에 대한 가중치.
    '''
    def __init__(self, weight_angle=10):
        '''
        EASTLoss 클래스를 초기화합니다.
        
        Args:
            weight_angle (float, optional): 각도 손실에 대한 가중치. 기본값은 10.
        '''
        super().__init__()
        self.weight_angle = weight_angle

    def forward(self, gt_score, pred_score, gt_geo, pred_geo, roi_mask):
        '''
        순전파 단계에서 손실을 계산합니다.
        
        Args:
            gt_score (torch.Tensor): 실제 스코어 맵 (Ground Truth).
            pred_score (torch.Tensor): 예측된 스코어 맵.
            gt_geo (torch.Tensor): 실제 지오메트리 맵 (Ground Truth).
            pred_geo (torch.Tensor): 예측된 지오메트리 맵.
            roi_mask (torch.Tensor): ROI 마스크.
        
        Returns:
            tuple: 총 손실 값과 개별 손실 구성 요소들 (cls_loss, angle_loss, iou_loss).
        '''
        # 실제 스코어가 하나도 없으면 모든 손실을 0으로 설정
        if torch.sum(gt_score) < 1:
            return torch.sum(pred_score + pred_geo) * 0, dict(cls_loss=None, angle_loss=None, iou_loss=None)

        # 분류 손실 계산 (Dice 손실)
        classify_loss = get_dice_loss(gt_score, pred_score * roi_mask)
        
        # 지오메트리 손실 계산 (IoU 손실 맵과 각도 손실 맵)
        iou_loss_map, angle_loss_map = get_geo_loss(gt_geo, pred_geo)

        # 각도 손실 계산 및 가중치 적용
        angle_loss = torch.sum(angle_loss_map * gt_score) / torch.sum(gt_score)
        angle_loss *= self.weight_angle
        
        # IoU 손실 계산
        iou_loss = torch.sum(iou_loss_map * gt_score) / torch.sum(gt_score)
        
        # 총 지오메트리 손실 계산
        geo_loss = angle_loss + iou_loss
        
        # 총 손실 계산
        total_loss = classify_loss + geo_loss

        return total_loss, dict(cls_loss=classify_loss.item(), angle_loss=angle_loss.item(),
                                iou_loss=iou_loss.item())
 angle_loss=angle_loss.item(),
                                iou_loss=iou_loss.item())
