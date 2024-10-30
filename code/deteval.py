import math
from collections import namedtuple
from copy import deepcopy

import numpy as np


def default_evaluation_params():
    """
    기본 평가 매개변수를 반환합니다.
    
    반환:
        dict: 검증 및 평가에 사용할 기본 매개변수 딕셔너리
    """
    return {
        'AREA_RECALL_CONSTRAINT': 0.8,           # 면적 재현율 제약 조건
        'AREA_PRECISION_CONSTRAINT': 0.4,        # 면적 정밀도 제약 조건
        'EV_PARAM_IND_CENTER_DIFF_THR': 1,       # 중심 차이 임계값
        'MTYPE_OO_O': 1.0,                        # 일대일 매칭 시 재현율 및 정밀도 가중치
        'MTYPE_OM_O': 0.8,                        # 일대다 매칭 시 재현율 가중치
        'MTYPE_OM_M': 1.0,                        # 일대다 매칭 시 정밀도 가중치
        'GT_SAMPLE_NAME_2_ID': 'gt_img_([0-9]+).txt',  # Ground Truth 샘플 이름 패턴
        'DET_SAMPLE_NAME_2_ID': 'res_img_([0-9]+).txt', # Detected 샘플 이름 패턴
        'CRLF': False                              # 라인이 Windows CRLF 형식으로 구분되는지 여부
    }


def calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict,
                         eval_hparams=None, bbox_format='rect', verbose=False):
    """
    감지된 바운딩 박스와 Ground Truth 바운딩 박스를 비교하여 평가 메트릭을 계산합니다.
    
    현재는 rect(xmin, ymin, xmax, ymax) 형식의 바운딩 박스만 지원하며, 다른 형식(quadrilateral,
    polygon 등)의 데이터가 들어오면 외접하는 rect로 변환하여 사용합니다.
    
    입력:
        pred_bboxes_dict (dict): 예측된 바운딩 박스 딕셔너리. 키는 샘플 이름, 값은 바운딩 박스 리스트.
        gt_bboxes_dict (dict): Ground Truth 바운딩 박스 딕셔너리. 키는 샘플 이름, 값은 바운딩 박스 리스트.
        eval_hparams (dict, optional): 평가에 사용할 매개변수 딕셔너리. 기본값은 None으로, default_evaluation_params() 사용.
        bbox_format (str, optional): 바운딩 박스 형식. 현재는 'rect'만 지원. 기본값은 'rect'.
        verbose (bool, optional): 상세 로그를 출력할지 여부. 기본값은 False.
    
    출력:
        dict: 평가 결과를 포함하는 딕셔너리
    """

    def one_to_one_match(row, col):
        """
        일대일 매칭을 확인합니다.
        
        입력:
            row (int): Ground Truth 바운딩 박스 인덱스
            col (int): Detected 바운딩 박스 인덱스
        
        출력:
            bool: 일대일 매칭이 성립하면 True, 아니면 False
        """
        cont = 0
        for j in range(len(recallMat[0])):
            if (recallMat[row, j] >= eval_hparams['AREA_RECALL_CONSTRAINT'] and 
                precisionMat[row, j] >= eval_hparams['AREA_PRECISION_CONSTRAINT']):
                cont += 1
        if cont != 1:
            return False
        cont = 0
        for i in range(len(recallMat)):
            if (recallMat[i, col] >= eval_hparams['AREA_RECALL_CONSTRAINT'] and 
                precisionMat[i, col] >= eval_hparams['AREA_PRECISION_CONSTRAINT']):
                cont += 1
        if cont != 1:
            return False

        if (recallMat[row, col] >= eval_hparams['AREA_RECALL_CONSTRAINT'] and 
            precisionMat[row, col] >= eval_hparams['AREA_PRECISION_CONSTRAINT']):
            return True
        return False

    def one_to_one_match_v2(row, col):
        """
        개선된 일대일 매칭 확인 함수.
        
        입력:
            row (int): Ground Truth 바운딩 박스 인덱스
            col (int): Detected 바운딩 박스 인덱스
        
        출력:
            bool: 일대일 매칭이 성립하면 True, 아니면 False
        """
        if row_sum[row] != 1:
            return False

        if col_sum[col] != 1:
            return False

        if (recallMat[row, col] >= eval_hparams['AREA_RECALL_CONSTRAINT'] and 
            precisionMat[row, col] >= eval_hparams['AREA_PRECISION_CONSTRAINT']):
            return True
        return False

    def num_overlaps_gt(gtNum):
        """
        특정 Ground Truth 바운딩 박스와 겹치는 Detected 바운딩 박스의 수를 계산합니다.
        
        입력:
            gtNum (int): Ground Truth 바운딩 박스 인덱스
        
        출력:
            int: 겹치는 Detected 바운딩 박스의 수
        """
        cont = 0
        for detNum in range(len(detRects)):
            if recallMat[gtNum, detNum] > 0:
                cont += 1
        return cont

    def num_overlaps_det(detNum):
        """
        특정 Detected 바운딩 박스와 겹치는 Ground Truth 바운딩 박스의 수를 계산합니다.
        
        입력:
            detNum (int): Detected 바운딩 박스 인덱스
        
        출력:
            int: 겹치는 Ground Truth 바운딩 박스의 수
        """
        cont = 0
        for gtNum in range(len(recallMat)):
            if recallMat[gtNum, detNum] > 0:
                cont += 1
        return cont

    def is_single_overlap(row, col):
        """
        특정 Ground Truth와 Detected 바운딩 박스가 단일으로 겹치는지 확인합니다.
        
        입력:
            row (int): Ground Truth 바운딩 박스 인덱스
            col (int): Detected 바운딩 박스 인덱스
        
        출력:
            bool: 단일 겹침이면 True, 아니면 False
        """
        if num_overlaps_gt(row) == 1 and num_overlaps_det(col) == 1:
            return True
        else:
            return False

    def one_to_many_match(gtNum):
        """
        하나의 Ground Truth 바운딩 박스가 여러 Detected 바운딩 박스와 매칭되는지 확인합니다.
        
        입력:
            gtNum (int): Ground Truth 바운딩 박스 인덱스
        
        출력:
            tuple: (매칭 여부 (bool), 매칭된 Detected 바운딩 박스 인덱스 리스트)
        """
        many_sum = 0
        detRects = []
        for detNum in range(len(recallMat[0])):
            if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0:
                if precisionMat[gtNum, detNum] >= eval_hparams['AREA_PRECISION_CONSTRAINT']:
                    many_sum += recallMat[gtNum, detNum]
                    detRects.append(detNum)
        if round(many_sum, 4) >= eval_hparams['AREA_RECALL_CONSTRAINT']:
            return True, detRects
        else:
            return False, []

    def many_to_one_match(detNum):
        """
        하나의 Detected 바운딩 박스가 여러 Ground Truth 바운딩 박스와 매칭되는지 확인합니다.
        
        입력:
            detNum (int): Detected 바운딩 박스 인덱스
        
        출력:
            tuple: (매칭 여부 (bool), 매칭된 Ground Truth 바운딩 박스 인덱스 리스트)
        """
        many_sum = 0
        gtRects = []
        for gtNum in range(len(recallMat)):
            if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0:
                if recallMat[gtNum, detNum] >= eval_hparams['AREA_RECALL_CONSTRAINT']:
                    many_sum += precisionMat[gtNum, detNum]
                    gtRects.append(gtNum)
        if round(many_sum, 4) >= eval_hparams['AREA_PRECISION_CONSTRAINT']:
            return True, gtRects
        else:
            return False, []

    def area(a, b):
        """
        두 사각형의 교집합 면적을 계산합니다.
        
        입력:
            a (Rectangle): 첫 번째 사각형
            b (Rectangle): 두 번째 사각형
        
        출력:
            float: 교집합 면적
        """
        dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin) + 1
        dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin) + 1
        if (dx >= 0) and (dy >= 0):
            return dx * dy
        else:
            return 0.0

    def center(r):
        """
        사각형의 중심점을 계산합니다.
        
        입력:
            r (Rectangle): 사각형
        
        출력:
            Point: 중심점
        """
        x = float(r.xmin) + float(r.xmax - r.xmin + 1) / 2.0
        y = float(r.ymin) + float(r.ymax - r.ymin + 1) / 2.0
        return Point(x, y)

    def point_distance(r1, r2):
        """
        두 점 사이의 유클리드 거리를 계산합니다.
        
        입력:
            r1 (Point): 첫 번째 점
            r2 (Point): 두 번째 점
        
        출력:
            float: 두 점 사이의 거리
        """
        distx = math.fabs(r1.x - r2.x)
        disty = math.fabs(r1.y - r2.y)
        return math.sqrt(distx * distx + disty * disty)

    def center_distance(r1, r2):
        """
        두 사각형의 중심점 사이의 거리를 계산합니다.
        
        입력:
            r1 (Rectangle): 첫 번째 사각형
            r2 (Rectangle): 두 번째 사각형
        
        출력:
            float: 중심점 사이의 거리
        """
        return point_distance(center(r1), center(r2))

    def diag(r):
        """
        사각형의 대각선 길이를 계산합니다.
        
        입력:
            r (Rectangle): 사각형
        
        출력:
            float: 대각선 길이
        """
        w = (r.xmax - r.xmin + 1)
        h = (r.ymax - r.ymin + 1)
        return math.sqrt(h * h + w * w)

    if eval_hparams is None:
        eval_hparams = default_evaluation_params()

    if bbox_format != 'rect':
        raise NotImplementedError("현재는 'rect' 형식의 바운딩 박스만 지원합니다.")

    # bbox들이 rect 형식이 아닌 경우 rect 형식으로 변환
    _pred_bboxes_dict, _gt_bboxes_dict = deepcopy(pred_bboxes_dict), deepcopy(gt_bboxes_dict)
    pred_bboxes_dict, gt_bboxes_dict = dict(), dict()
    for sample_name, bboxes in _pred_bboxes_dict.items():
        # 원래 rect 형식이었으면 변환 없이 그대로 이용
        if len(bboxes) > 0 and np.array(bboxes[0]).ndim == 1 and len(bboxes[0]) == 4:
            pred_bboxes_dict = _pred_bboxes_dict
            break

        pred_bboxes_dict[sample_name] = []
        for bbox in map(np.array, bboxes):
            rect = [bbox[:, 0].min(), bbox[:, 1].min(), bbox[:, 0].max(), bbox[:, 1].max()]
            pred_bboxes_dict[sample_name].append(rect)
    for sample_name, bboxes in _gt_bboxes_dict.items():
        # 원래 rect 형식이었으면 변환 없이 그대로 이용
        if len(bboxes) > 0 and np.array(bboxes[0]).ndim == 1 and len(bboxes[0]) == 4:
            gt_bboxes_dict = _gt_bboxes_dict
            break

        gt_bboxes_dict[sample_name] = []
        for bbox in map(np.array, bboxes):
            rect = [bbox[:, 0].min(), bbox[:, 1].min(), bbox[:, 0].max(), bbox[:, 1].max()]
            gt_bboxes_dict[sample_name].append(rect)

    perSampleMetrics = {}

    methodRecallSum = 0
    methodPrecisionSum = 0

    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    Point = namedtuple('Point', 'x y')

    numGt = 0
    numDet = 0

    for sample_name in gt_bboxes_dict:

        recall = 0
        precision = 0
        hmean = 0
        recallAccum = 0.0
        precisionAccum = 0.0
        gtRects = []
        detRects = []
        gtPolPoints = []
        detPolPoints = []
        pairs = []
        evaluationLog = ""

        recallMat = np.empty([1, 1])
        precisionMat = np.empty([1, 1])

        pointsList = gt_bboxes_dict[sample_name]

        for n in range(len(pointsList)):
            points = pointsList[n]
            gtRect = Rectangle(*points)
            gtRects.append(gtRect)
            gtPolPoints.append(np.array(points).tolist())

        evaluationLog += "GT rectangles: " + str(len(gtRects)) + '\n'

        if sample_name in pred_bboxes_dict:
            pointsList = pred_bboxes_dict[sample_name]

            for n in range(len(pointsList)):
                points = pointsList[n]
                detRect = Rectangle(*points)
                detRects.append(detRect)
                detPolPoints.append(np.array(points).tolist())

            evaluationLog += "DET rectangles: " + str(len(detRects)) + '\n'

            if len(gtRects) == 0:
                recall = 1.0
                precision = 0.0 if len(detRects) > 0 else 1.0

            if len(detRects) > 0:
                # 재현율과 정밀도 매트릭스 계산
                outputShape = [len(gtRects), len(detRects)]
                recallMat = np.empty(outputShape)
                precisionMat = np.empty(outputShape)
                gtRectMat = np.zeros(len(gtRects), np.int8)
                detRectMat = np.zeros(len(detRects), np.int8)
                for gtNum in range(len(gtRects)):
                    for detNum in range(len(detRects)):
                        rG = gtRects[gtNum]
                        rD = detRects[detNum]
                        intersected_area = area(rG, rD)
                        rgDimensions = ((rG.xmax - rG.xmin + 1) * (rG.ymax - rG.ymin + 1))
                        rdDimensions = ((rD.xmax - rD.xmin + 1) * (rD.ymax - rD.ymin + 1))
                        recallMat[gtNum, detNum] = 0 if rgDimensions == 0 else intersected_area / rgDimensions
                        precisionMat[gtNum, detNum] = 0 if rdDimensions == 0 else intersected_area / rdDimensions

                recall_cond = recallMat >= eval_hparams['AREA_RECALL_CONSTRAINT']
                precision_cond = precisionMat >= eval_hparams['AREA_PRECISION_CONSTRAINT']
                cond = recall_cond & precision_cond
                col_sum = np.sum(cond, axis=0)
                row_sum = np.sum(cond, axis=1)

                # 일대일 매칭 찾기
                evaluationLog += "Find one-to-one matches\n"
                for gtNum in range(len(gtRects)):
                    for detNum in range(len(detRects)):
                        if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0:
                            match = one_to_one_match_v2(gtNum, detNum)
                            if match is True:
                                # 매칭을 일대일로 검증
                                if is_single_overlap(gtNum, detNum) is True:
                                    rG = gtRects[gtNum]
                                    rD = detRects[detNum]
                                    normDist = center_distance(rG, rD)
                                    normDist /= diag(rG) + diag(rD)
                                    normDist *= 2.0
                                    if normDist < eval_hparams['EV_PARAM_IND_CENTER_DIFF_THR']:
                                        gtRectMat[gtNum] = 1
                                        detRectMat[detNum] = 1
                                        recallAccum += eval_hparams['MTYPE_OO_O']
                                        precisionAccum += eval_hparams['MTYPE_OO_O']
                                        pairs.append({'gt': gtNum, 'det': detNum, 'type': 'OO'})
                                        evaluationLog += "Match GT #" + str(gtNum) + " with Det #" + str(detNum) + "\n"
                                    else:
                                        evaluationLog += "Match Discarded GT #" + str(gtNum) + " with Det #" + str(detNum) + " normDist: " + str(normDist) + " \n"
                                else:
                                    evaluationLog += "Match Discarded GT #" + str(gtNum) + " with Det #" + str(detNum) + " not single overlap\n"

                # 일대다 매칭 찾기
                evaluationLog += "Find one-to-many matches\n"
                for gtNum in range(len(gtRects)):
                    match, matchesDet = one_to_many_match(gtNum)
                    if match is True:
                        evaluationLog += "num_overlaps_gt=" + str(num_overlaps_gt(gtNum))
                        # 매칭을 일대일로 검증
                        if num_overlaps_gt(gtNum) >= 2:
                            gtRectMat[gtNum] = 1
                            recallAccum += (eval_hparams['MTYPE_OO_O'] if len(matchesDet) == 1 else eval_hparams['MTYPE_OM_O'])
                            precisionAccum += (eval_hparams['MTYPE_OO_O'] if len(matchesDet) == 1 else eval_hparams['MTYPE_OM_O'] * len(matchesDet))
                            pairs.append({'gt': gtNum, 'det': matchesDet, 'type': 'OO' if len(matchesDet) == 1 else 'OM'})
                            for detNum in matchesDet:
                                detRectMat[detNum] = 1
                            evaluationLog += "Match GT #" + str(gtNum) + " with Det #" + str(matchesDet) + "\n"
                        else:
                            evaluationLog += "Match Discarded GT #" + str(gtNum) + " with Det #" + str(matchesDet) + " not single overlap\n"

                # 다대일 매칭 찾기
                evaluationLog += "Find many-to-one matches\n"
                for detNum in range(len(detRects)):
                    match, matchesGt = many_to_one_match(detNum)
                    if match is True:
                        # 매칭을 다대일로 검증
                        if num_overlaps_det(detNum) >= 2:
                            detRectMat[detNum] = 1
                            recallAccum += (eval_hparams['MTYPE_OO_O'] if len(matchesGt) == 1 else eval_hparams['MTYPE_OM_M'] * len(matchesGt))
                            precisionAccum += (eval_hparams['MTYPE_OO_O'] if len(matchesGt) == 1 else eval_hparams['MTYPE_OM_M'])
                            pairs.append({'gt': matchesGt, 'det': detNum, 'type': 'OO' if len(matchesGt) == 1 else 'MO'})
                            for gtNum in matchesGt:
                                gtRectMat[gtNum] = 1
                            evaluationLog += "Match GT #" + str(matchesGt) + " with Det #" + str(detNum) + "\n"
                        else:
                            evaluationLog += "Match Discarded GT #" + str(matchesGt) + " with Det #" + str(detNum) + " not single overlap\n"

                numGtCare = len(gtRects)
                if numGtCare == 0:
                    recall = float(1)
                    precision = float(0) if len(detRects) > 0 else float(1)
                else:
                    recall = float(recallAccum) / numGtCare
                    precision = float(0) if len(detRects) == 0 else float(precisionAccum) / len(detRects)
                hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)

        methodRecallSum += recallAccum
        methodPrecisionSum += precisionAccum
        numGt += len(gtRects)
        numDet += len(detRects)

        perSampleMetrics[sample_name] = {
            'precision': precision,
            'recall': recall,
            'hmean': hmean,
            'pairs': pairs,
            'recall_matrix': [] if len(detRects) > 100 else recallMat.tolist(),
            'precision_matrix': [] if len(detRects) > 100 else precisionMat.tolist(),
            'gt_bboxes': gtPolPoints,
            'det_bboxes': detPolPoints,
        }

        if verbose:
            perSampleMetrics[sample_name].update(evaluation_log=evaluationLog)

    methodRecall = 0 if numGt == 0 else methodRecallSum / numGt
    methodPrecision = 0 if numDet == 0 else methodPrecisionSum / numDet
    methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * methodRecall * methodPrecision / (methodRecall + methodPrecision)

    methodMetrics = {'precision': methodPrecision, 'recall': methodRecall, 'hmean': methodHmean}

    resDict = {
        'calculated': True,
        'Message': '',
        'total': methodMetrics,
        'per_sample': perSampleMetrics,
        'eval_hparams': eval_hparams
    }

    return resDict

