import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss import EASTLoss


# VGG 네트워크 설정을 위한 구성 리스트
cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


def make_layers(cfg, batch_norm=False):
    """
    VGG 네트워크의 레이어를 생성하는 함수입니다.
    
    입력:
        cfg (리스트): 각 레이어의 구성 요소를 정의하는 리스트. 숫자는 Conv2d의 출력 채널 수를 의미하며, 'M'은 MaxPooling 레이어를 의미합니다.
        batch_norm (bool, optional): 배치 정규화를 사용할지 여부. 기본값은 False.
    
    출력:
        nn.Sequential: 생성된 레이어들을 포함하는 시퀀셜 모델
    """
    layers = []
    in_channels = 3  # 입력 채널 수 (RGB 이미지)
    for v in cfg:
        if v == 'M':
            # MaxPooling 레이어 추가
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # Conv2d 레이어 추가
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                # 배치 정규화과 ReLU 활성화 함수 추가
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                # ReLU 활성화 함수 추가
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v  # 다음 레이어의 입력 채널 수 업데이트
    return nn.Sequential(*layers)


class VGG(nn.Module):
    """
    VGG 네트워크 클래스입니다.
    
    이 클래스는 VGG 네트워크의 특성을 추출하고 분류기를 포함합니다.
    """
    def __init__(self, features):
        """
        VGG 클래스의 초기화 함수입니다.
        
        입력:
            features (nn.Sequential): VGG의 특성 추출 레이어
        """
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # 평균 풀링 레이어
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # 완전 연결층
            nn.ReLU(True),
            nn.Dropout(),  # 드롭아웃 레이어
            nn.Linear(4096, 4096),  # 두 번째 완전 연결층
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),  # 최종 출력층 (클래스 수에 따라 변경 가능)
        )

        # 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Conv2d 레이어의 가중치를 He 초기화
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # BatchNorm2d 레이어의 가중치와 편향 초기화
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Linear 레이어의 가중치와 편향 초기화
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        순전파 함수입니다.
        
        입력:
            x (torch.Tensor): 입력 이미지 텐서
        
        출력:
            torch.Tensor: 네트워크의 최종 출력
        """
        x = self.features(x)  # 특성 추출 레이어 통과
        x = self.avgpool(x)    # 평균 풀링 레이어 통과
        x = x.view(x.size(0), -1)  # 텐서 평탄화
        x = self.classifier(x)      # 분류기 통과
        return x


# EAST 논문 Fig 3. 특성 추출기 (VGG 기반)
class Extractor(nn.Module):
    """
    EAST 모델의 특성 추출기 클래스입니다.
    
    VGG 네트워크를 기반으로 특성을 추출하며, 사전 학습된 가중치를 로드할 수 있습니다.
    """
    def __init__(self, pretrained):
        """
        Extractor 클래스의 초기화 함수입니다.
        
        입력:
            pretrained (bool): 사전 학습된 가중치를 사용할지 여부
        """
        super().__init__()
        vgg16_bn = VGG(make_layers(cfg, batch_norm=True))  # 배치 정규화를 사용하는 VGG 생성
        if pretrained:
            # 사전 학습된 가중치를 로드
            vgg16_bn.load_state_dict(torch.load('./pths/vgg16_bn-6c64b313.pth'))
        self.features = vgg16_bn.features  # VGG의 특성 추출 레이어만 사용

    def forward(self, x):
        """
        순전파 함수입니다.
        
        입력:
            x (torch.Tensor): 입력 이미지 텐서
        
        출력:
            list of torch.Tensor: 각 MaxPooling 레이어 이후의 출력 리스트
        """
        out = []
        for m in self.features:
            x = m(x)
            if isinstance(m, nn.MaxPool2d):
                out.append(x)
        return out


# EAST 논문 Fig 3. 특성 병합 분기
class Merge(nn.Module):
    """
    EAST 모델의 특성 병합 분기 클래스입니다.
    
    여러 특성 맵을 병합하여 최종적인 지오메트리 정보를 생성합니다.
    """
    def __init__(self):
        """
        Merge 클래스의 초기화 함수입니다.
        """
        super().__init__()

        # 첫 번째 병합 단계
        self.conv1 = nn.Conv2d(1024, 128, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        # 두 번째 병합 단계
        self.conv3 = nn.Conv2d(384, 64, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        # 세 번째 병합 단계
        self.conv5 = nn.Conv2d(192, 32, 1)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU()

        # 네 번째 병합 단계
        self.conv7 = nn.Conv2d(96, 32, 1)
        self.bn7 = nn.BatchNorm2d(32)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(32)
        self.relu8 = nn.ReLU()

        # 다섯 번째 병합 단계
        self.conv9 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn9 = nn.BatchNorm2d(32)
        self.relu9 = nn.ReLU()

        # 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Conv2d 레이어의 가중치를 He 초기화
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # BatchNorm2d 레이어의 가중치와 편향 초기화
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        순전파 함수입니다.
        
        입력:
            x (list of torch.Tensor): Extractor에서 나온 여러 특성 맵 리스트
        
        출력:
            torch.Tensor: 병합된 특성 맵
        """
        # 첫 번째 병합 단계
        y = F.interpolate(x[4], scale_factor=2, mode='bilinear', align_corners=True)  # 특성 맵 업샘플링
        y = torch.cat((y, x[3]), 1)  # 업샘플링된 특성 맵과 이전 특성 맵을 채널 차원에서 결합
        y = self.relu1(self.bn1(self.conv1(y)))
        y = self.relu2(self.bn2(self.conv2(y)))

        # 두 번째 병합 단계
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        y = torch.cat((y, x[2]), 1)
        y = self.relu3(self.bn3(self.conv3(y)))
        y = self.relu4(self.bn4(self.conv4(y)))

        # 세 번째 병합 단계
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        y = torch.cat((y, x[1]), 1)
        y = self.relu5(self.bn5(self.conv5(y)))
        y = self.relu6(self.bn6(self.conv6(y)))

        # 네 번째 병합 단계
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        y = torch.cat((y, x[0]), 1)
        y = self.relu7(self.bn7(self.conv7(y)))
        y = self.relu8(self.bn8(self.conv8(y)))

        # 다섯 번째 병합 단계
        y = self.relu9(self.bn9(self.conv9(y)))

        return y


# EAST 논문 Fig 3. 출력 레이어
class Output(nn.Module):
    """
    EAST 모델의 출력 레이어 클래스입니다.
    
    이 클래스는 스코어 맵, 지오메트리 맵(위치 및 각도)을 생성합니다.
    """
    def __init__(self, scope=512):
        """
        Output 클래스의 초기화 함수입니다.
        
        입력:
            scope (int, optional): 지오메트리 맵의 범위 스케일. 기본값은 512.
        """
        super().__init__()
        # 스코어 맵을 생성하는 Conv2d 레이어와 시그모이드 활성화 함수
        self.conv1 = nn.Conv2d(32, 1, 1)
        self.sigmoid1 = nn.Sigmoid()
        # 지오메트리 맵을 생성하는 Conv2d 레이어와 시그모이드 활성화 함수
        self.conv2 = nn.Conv2d(32, 4, 1)
        self.sigmoid2 = nn.Sigmoid()
        # 각도 맵을 생성하는 Conv2d 레이어와 시그모이드 활성화 함수
        self.conv3 = nn.Conv2d(32, 1, 1)
        self.sigmoid3 = nn.Sigmoid()
        self.scope = scope  # 지오메트리 맵의 범위 스케일

        # 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Conv2d 레이어의 가중치를 He 초기화
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        순전파 함수입니다.
        
        입력:
            x (torch.Tensor): Merge에서 병합된 특성 맵
        
        출력:
            tuple: 스코어 맵, 지오메트리 맵 (위치, 각도)
        """
        score = self.sigmoid1(self.conv1(x))  # 스코어 맵 생성
        loc = self.sigmoid2(self.conv2(x)) * self.scope  # 지오메트리 맵(위치) 생성 및 스케일 조정
        angle = (self.sigmoid3(self.conv3(x)) - 0.5) * math.pi  # 각도 맵 생성 ([-π/2, π/2] 범위)
        geo = torch.cat((loc, angle), 1)  # 위치와 각도 맵을 채널 차원에서 결합
        return score, geo


class EAST(nn.Module):
    """
    EAST (Efficient and Accurate Scene Text Detector) 모델 클래스입니다.
    
    이 클래스는 특성 추출기, 특성 병합 분기, 출력 레이어로 구성되어 있으며,
    손실 함수도 포함하고 있습니다.
    """
    def __init__(self, pretrained=True):
        """
        EAST 클래스의 초기화 함수입니다.
        
        입력:
            pretrained (bool, optional): 특성 추출기에 사전 학습된 가중치를 사용할지 여부. 기본값은 True.
        """
        super(EAST, self).__init__()
        self.extractor = Extractor(pretrained)  # 특성 추출기 초기화
        self.merge = Merge()                     # 특성 병합 분기 초기화
        self.output = Output()                   # 출력 레이어 초기화

        self.criterion = EASTLoss()              # 손실 함수 초기화

    def forward(self, x):
        """
        순전파 함수입니다.
        
        입력:
            x (torch.Tensor): 입력 이미지 텐서
        
        출력:
            tuple: 스코어 맵과 지오메트리 맵
        """
        features = self.extractor(x)  # 특성 추출기 통과
        merged = self.merge(features) # 특성 병합 분기 통과
        score, geo = self.output(merged)  # 출력 레이어 통과
        return score, geo

    def train_step(self, image, score_map, geo_map, roi_mask):
        """
        학습 단계에서 손실을 계산하는 함수입니다.
        
        입력:
            image (torch.Tensor): 입력 이미지 텐서
            score_map (torch.Tensor): Ground Truth 스코어 맵
            geo_map (torch.Tensor): Ground Truth 지오메트리 맵
            roi_mask (torch.Tensor): ROI 마스크
        
        출력:
            tuple:
                loss (torch.Tensor): 총 손실 값
                extra_info (dict): 개별 손실 값과 예측 맵 정보
        """
        device = list(self.parameters())[0].device  # 모델이 있는 디바이스 확인
        # 입력 데이터를 디바이스로 이동
        image, score_map, geo_map, roi_mask = (
            image.to(device), score_map.to(device),
            geo_map.to(device), roi_mask.to(device)
        )
        # 순전파를 통해 예측 맵 생성
        pred_score_map, pred_geo_map = self.forward(image)

        # 손실 계산
        loss, values_dict = self.criterion(score_map, pred_score_map, geo_map, pred_geo_map, roi_mask)
        
        # 추가 정보에 손실 값을 포함
        extra_info = dict(**values_dict, score_map=pred_score_map, geo_map=pred_geo_map)

        return loss, extra_info


if __name__ == '__main__':
    """
    테스트용 메인 블록입니다.
    
    EAST 모델을 인스턴스화하고, 임의의 입력을 통해 스코어 맵과 지오메트리 맵의 크기를 확인합니다.
    """
    m = EAST()
    x = torch.randn(1, 3, 1024, 1024)  # 임의의 입력 이미지 텐서 생성
    score, geo = m(x)                   # 모델을 통해 예측
    print(score.shape)                  # 스코어 맵의 형태 출력
    print(geo.shape)                    # 지오메트리 맵의 형태 출력
