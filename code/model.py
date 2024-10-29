import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss import EASTLoss


# VGG 네트워크의 구성 요소를 정의하는 설정 리스트
cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


def make_layers(cfg, batch_norm=False):
    '''
    VGG 네트워크의 레이어를 생성합니다.
    
    Args:
        cfg (list): 네트워크 구성 설정 리스트. 숫자는 Conv 레이어의 출력 채널 수, 'M'은 MaxPooling을 의미합니다.
        batch_norm (bool, optional): 배치 정규화를 사용할지 여부. 기본값은 False.
    
    Returns:
        nn.Sequential: 생성된 레이어들의 순차적 모음.
    '''
    layers = []
    in_channels = 3  # 입력 이미지의 채널 수 (RGB 이미지이므로 3)
    for v in cfg:
        if v == 'M':
            # MaxPooling 레이어 추가
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # Conv 레이어 추가
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                # 배치 정규화과 ReLU 활성화 함수 추가
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                # ReLU 활성화 함수만 추가
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v  # 다음 Conv 레이어의 입력 채널 수 업데이트
    return nn.Sequential(*layers)


class VGG(nn.Module):
    '''
    VGG 네트워크 클래스.
    
    Attributes:
        features (nn.Sequential): VGG 네트워크의 특징 추출 레이어.
        avgpool (nn.AdaptiveAvgPool2d): Adaptive Average Pooling 레이어.
        classifier (nn.Sequential): VGG 네트워크의 분류기 레이어.
    '''
    def __init__(self, features):
        '''
        VGG 클래스를 초기화합니다.
        
        Args:
            features (nn.Sequential): VGG 네트워크의 특징 추출 레이어.
        '''
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # 출력 크기를 (7, 7)로 조정
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # 완전 연결층
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
        )

        # 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming 정규분포로 가중치 초기화
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # 배치 정규화의 가중치와 편향 초기화
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 선형층의 가중치와 편향 초기화
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        '''
        순전파를 정의합니다.
        
        Args:
            x (torch.Tensor): 입력 이미지 텐서.
        
        Returns:
            torch.Tensor: 분류 결과.
        '''
        x = self.features(x)  # 특징 추출
        x = self.avgpool(x)    # 평균 풀링
        x = x.view(x.size(0), -1)  # 평탄화
        x = self.classifier(x)  # 분류기 통과
        return x


# EAST 논문 Fig 3. 특징 추출기 스템 (VGG 기반)
class Extractor(nn.Module):
    '''
    EAST 모델의 특징 추출기 클래스.
    
    Attributes:
        features (nn.Sequential): VGG 기반의 특징 추출 레이어.
    '''
    def __init__(self, pretrained):
        '''
        Extractor 클래스를 초기화합니다.
        
        Args:
            pretrained (bool): 사전 학습된 가중치를 사용할지 여부.
        '''
        super().__init__()
        vgg16_bn = VGG(make_layers(cfg, batch_norm=True))  # VGG16 with BatchNorm
        if pretrained:
            # 사전 학습된 VGG16 모델의 가중치 로드
            vgg16_bn.load_state_dict(torch.load('./pths/vgg16_bn-6c64b313.pth'))
        self.features = vgg16_bn.features  # 특징 추출 레이어 설정

    def forward(self, x):
        '''
        순전파를 정의합니다.
        
        Args:
            x (torch.Tensor): 입력 이미지 텐서.
        
        Returns:
            list of torch.Tensor: 각 MaxPooling 레이어의 출력.
        '''
        out = []
        for m in self.features:
            x = m(x)
            if isinstance(m, nn.MaxPool2d):
                out.append(x)  # MaxPooling 레이어의 출력 저장
        return out


# EAST 논문 Fig 3. 특징 병합 분기
class Merge(nn.Module):
    '''
    EAST 모델의 특징 병합 분기 클래스.
    
    Attributes:
        conv1, conv2, ..., conv9 (nn.Conv2d): 다양한 Conv 레이어.
        bn1, bn2, ..., bn9 (nn.BatchNorm2d): 배치 정규화 레이어.
        relu1, relu2, ..., relu9 (nn.ReLU): ReLU 활성화 함수.
    '''
    def __init__(self):
        '''
        Merge 클래스를 초기화합니다.
        '''
        super().__init__()

        # 첫 번째 블록
        self.conv1 = nn.Conv2d(1024, 128, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        # 두 번째 블록
        self.conv3 = nn.Conv2d(384, 64, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        # 세 번째 블록
        self.conv5 = nn.Conv2d(192, 32, 1)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU()

        # 네 번째 블록
        self.conv7 = nn.Conv2d(96, 32, 1)
        self.bn7 = nn.BatchNorm2d(32)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(32)
        self.relu8 = nn.ReLU()

        # 마지막 블록
        self.conv9 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn9 = nn.BatchNorm2d(32)
        self.relu9 = nn.ReLU()

        # 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming 정규분포로 가중치 초기화
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # 배치 정규화의 가중치와 편향 초기화
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        '''
        순전파를 정의합니다.
        
        Args:
            x (list of torch.Tensor): 특징 추출기에서 나온 여러 레이어의 출력.
        
        Returns:
            torch.Tensor: 병합된 특징 맵.
        '''
        # x는 특징 추출기의 여러 MaxPooling 레이어 출력 리스트
        y = F.interpolate(x[4], scale_factor=2, mode='bilinear', align_corners=True)  # 업샘플링
        y = torch.cat((y, x[3]), 1)  # 해당 레이어와 결합
        y = self.relu1(self.bn1(self.conv1(y)))  # Conv1-BN1-ReLU1
        y = self.relu2(self.bn2(self.conv2(y)))  # Conv2-BN2-ReLU2

        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)  # 업샘플링
        y = torch.cat((y, x[2]), 1)  # 해당 레이어와 결합
        y = self.relu3(self.bn3(self.conv3(y)))  # Conv3-BN3-ReLU3
        y = self.relu4(self.bn4(self.conv4(y)))  # Conv4-BN4-ReLU4

        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)  # 업샘플링
        y = torch.cat((y, x[1]), 1)  # 해당 레이어와 결합
        y = self.relu5(self.bn5(self.conv5(y)))  # Conv5-BN5-ReLU5
        y = self.relu6(self.bn6(self.conv6(y)))  # Conv6-BN6-ReLU6

        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)  # 업샘플링
        y = torch.cat((y, x[0]), 1)  # 해당 레이어와 결합
        y = self.relu7(self.bn7(self.conv7(y)))  # Conv7-BN7-ReLU7
        y = self.relu8(self.bn8(self.conv8(y)))  # Conv8-BN8-ReLU8

        y = self.relu9(self.bn9(self.conv9(y)))  # Conv9-BN9-ReLU9

        return y


# EAST 논문 Fig 3. 출력 레이어
class Output(nn.Module):
    '''
    EAST 모델의 출력 레이어 클래스.
    
    Attributes:
        conv1 (nn.Conv2d): 스코어 맵을 생성하는 Conv 레이어.
        sigmoid1 (nn.Sigmoid): 스코어 맵에 시그모이드 활성화 함수 적용.
        conv2 (nn.Conv2d): 지오메트리 맵을 생성하는 Conv 레이어.
        sigmoid2 (nn.Sigmoid): 지오메트리 맵에 시그모이드 활성화 함수 적용.
        conv3 (nn.Conv2d): 각도 맵을 생성하는 Conv 레이어.
        sigmoid3 (nn.Sigmoid): 각도 맵에 시그모이드 활성화 함수 적용.
        scope (int): 지오메트리 맵의 스케일을 조정하기 위한 상수.
    '''
    def __init__(self, scope=512):
        '''
        Output 클래스를 초기화합니다.
        
        Args:
            scope (int, optional): 지오메트리 맵의 스케일을 조정하기 위한 상수. 기본값은 512.
        '''
        super().__init__()
        self.conv1 = nn.Conv2d(32, 1, 1)  # 스코어 맵 생성 Conv 레이어
        self.sigmoid1 = nn.Sigmoid()       # 시그모이드 활성화 함수
        self.conv2 = nn.Conv2d(32, 4, 1)   # 지오메트리 맵 생성 Conv 레이어
        self.sigmoid2 = nn.Sigmoid()       # 시그모이드 활성화 함수
        self.conv3 = nn.Conv2d(32, 1, 1)   # 각도 맵 생성 Conv 레이어
        self.sigmoid3 = nn.Sigmoid()       # 시그모이드 활성화 함수
        self.scope = scope

        # 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming 정규분포로 가중치 초기화
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        '''
        순전파를 정의합니다.
        
        Args:
            x (torch.Tensor): Merge 클래스에서 나온 특징 맵.
        
        Returns:
            tuple:
                - torch.Tensor: 스코어 맵.
                - torch.Tensor: 지오메트리 맵 (d1, d2, d3, d4, angle).
        '''
        score = self.sigmoid1(self.conv1(x))  # 스코어 맵 생성 및 활성화
        loc   = self.sigmoid2(self.conv2(x)) * self.scope  # 지오메트리 맵 생성, 활성화 및 스케일 조정
        angle = (self.sigmoid3(self.conv3(x)) - 0.5) * math.pi  # 각도 맵 생성, 활성화 및 변환
        geo   = torch.cat((loc, angle), 1)  # 지오메트리 맵과 각도 맵을 결합
        return score, geo


class EAST(nn.Module):
    '''
    EAST 모델 클래스.
    
    Attributes:
        extractor (Extractor): 특징 추출기.
        merge (Merge): 특징 병합 분기.
        output (Output): 출력 레이어.
        criterion (EASTLoss): 손실 함수.
    '''
    def __init__(self, pretrained=True):
        '''
        EAST 클래스를 초기화합니다.
        
        Args:
            pretrained (bool, optional): 사전 학습된 가중치를 사용할지 여부. 기본값은 True.
        '''
        super(EAST, self).__init__()
        self.extractor = Extractor(pretrained)  # 특징 추출기 초기화
        self.merge = Merge()                     # 특징 병합 분기 초기화
        self.output = Output()                   # 출력 레이어 초기화

        self.criterion = EASTLoss()              # 손실 함수 초기화

    def forward(self, x):
        '''
        순전파를 정의합니다.
        
        Args:
            x (torch.Tensor): 입력 이미지 텐서.
        
        Returns:
            tuple:
                - torch.Tensor: 스코어 맵.
                - torch.Tensor: 지오메트리 맵.
        '''
        return self.output(self.merge(self.extractor(x)))  # 특징 추출 -> 병합 -> 출력

    def train_step(self, image, score_map, geo_map, roi_mask):
        '''
        학습 단계에서 손실을 계산하고 정보를 반환합니다.
        
        Args:
            image (torch.Tensor): 입력 이미지 텐서.
            score_map (torch.Tensor): 실제 스코어 맵 (Ground Truth).
            geo_map (torch.Tensor): 실제 지오메트리 맵 (Ground Truth).
            roi_mask (torch.Tensor): ROI 마스크.
        
        Returns:
            tuple:
                - torch.Tensor: 총 손실 값.
                - dict: 개별 손실 구성 요소들과 예측된 스코어 맵, 지오메트리 맵.
        '''
        device = list(self.parameters())[0].device  # 모델이 있는 디바이스 확인
        # 모든 입력을 해당 디바이스로 이동
        image, score_map, geo_map, roi_mask = (image.to(device), score_map.to(device),
                                               geo_map.to(device), roi_mask.to(device))
        pred_score_map, pred_geo_map = self.forward(image)  # 순전파를 통해 예측된 스코어 맵과 지오메트리 맵 얻기

        # 손실 계산
        loss, values_dict = self.criterion(score_map, pred_score_map, geo_map, pred_geo_map,
                                           roi_mask)
        # 추가 정보 저장
        extra_info = dict(**values_dict, score_map=pred_score_map, geo_map=pred_geo_map)

        return loss, extra_info


if __name__ == '__main__':
    '''
    EAST 모델을 테스트하기 위한 메인 블록.
    '''
    m = EAST()  # EAST 모델 인스턴스 생성
    x = torch.randn(1, 3, 1024, 1024)  # 임의의 입력 텐서 생성 (배치 크기 1, 채널 3, 크기 1024x1024)
    score, geo = m(x)  # 모델에 입력하여 스코어 맵과 지오메트리 맵 얻기
    print(score.shape)  # 스코어 맵의 형태 출력
    print(geo.shape)    # 지오메트리 맵의 형태 출력
