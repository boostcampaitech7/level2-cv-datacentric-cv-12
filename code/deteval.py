import math
from collections import namedtuple
from copy import deepcopy

import numpy as np


def default_evaluation_params():
    """
    기본 평가 매개변수: 검증 및 평가에 사용할 기본 매개변수를 반환합니다.

    반환값:
        dict: 평가에 사용될 기본 매개변수들을 포함한 사전.
            - 'AREA_RECALL_CONSTRAINT': 면적 기반 재현율 제약 조건.
            - 'AREA_PRECISION_CONSTRAINT': 면적 기반 정밀도 제약 조건.
            - 'EV_PARAM_IND_CENTER_DIFF_THR': 중심점 차이 임계값.
            - 'MTYPE_OO_O': 일대일 매칭 시 가중치.
            - 'MTYPE_OM_O': 일대다 매칭 시 가중치.
            - 'MTYPE_OM_M': 다대일 매칭 시 가중치.
            - 'GT_SAMPLE_NAME_2_ID': Ground Truth 샘플 이름에서 ID를 추출하는 정규식 패턴.
            - 'DET_SAMPLE_NAME_2_ID': Detection 샘플 이름에서 ID를 추출하는 정규식 패턴.
            - 'CRLF': 줄 구분이 Windows CRLF 형식인지 여부.
    """
    return {
        'AREA_RECALL_CONSTRAINT': 0.8,
        'AREA_PRECISION_CONSTRAINT': 0.4,
        'EV_PARAM_IND_CENTER_DIFF_THR': 1,
        'MTYPE_OO_O': 1.0,
        'MTYPE_OM_O': 0.8,
        'MTYPE_OM_M': 1.0,
        'GT_SAMPLE_NAME_2_ID': 'gt_img_([0-9]+).txt',
        'DET_SAMPLE_NAME_2_ID': 'res_img_([0-9]+).txt',
        'CRLF': False  # 줄이 Windows CRLF 형식으로 구분되는지 여부
    }


def calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict,
                         eval_hparams=None, bbox_format='rect', verbose=False):
    """
    검출 평가 지표(정밀도, 재현율, 조화 평균)를 예측된 바운딩 박스와 실제 바운딩 박스를 비교하여 계산합니다.

    현재는 직사각형(xmin, ymin, xmax, ymax) 형식의 바운딩 박스만 지원합니다. 다른 형식(사변형, 다각형 등)의 데이터가 들어오면 외접하는 직사각형으로 변환하여 이용합니다.

    Args:
        pred_bboxes_dict (dict): 샘플 이름을 키로 하고 예측된 바운딩 박스 리스트를 값으로 하는 사전.
        gt_bboxes_dict (dict): 샘플 이름을 키로 하고 실제 바운딩 박스 리스트를 값으로 하는 사전.
        eval_hparams (dict, optional): 평가에 사용할 매개변수. None인 경우 기본 매개변수를 사용합니다.
        bbox_format (str, optional): 바운딩 박스의 형식. 현재는 'rect'만 지원합니다.
        verbose (bool, optional): True인 경우 상세 평가 로그를 결과에 포함합니다.

    Returns:
        dict: 전체 지표, 샘플별 지표, 평가 매개변수를 포함하는 사전.
            - 'calculated' (bool): 지표가 성공적으로 계산되었는지 여부.
            - 'Message' (str): 계산 과정과 관련된 메시지.
            - 'total' (dict): 전체 정밀도, 재현율, 조화 평균.
            - 'per_sample' (dict): 각 샘플별 정밀도, 재현율, 조화 평균 및 기타 정보.
            - 'eval_hparams' (dict): 사용된 평가 매개변수.
    """

    # 헬퍼 함수들

    def one_to_one_match(row, col):
        """
        주어진 행(row)과 열(col)이 일대일 매칭 조건을 충족하는지 확인합니다.

        Args:
            row (int): 실제 바운딩 박스의 인덱스.
            col (int): 예측 바운딩 박스의 인덱스.

        Returns:
            bool: 일대일 매칭 조건을 만족하면 True, 그렇지 않으면 False.
        """
        cont = 0
        for j in range(len(recallMat[0])):
            if recallMat[row, j] >= eval_hparams['AREA_RECALL_CONSTRAINT'] and \
               precisionMat[row, j] >= eval_hparams['AREA_PRECISION_CONSTRAINT']:
                cont += 1
        if cont != 1:
            return False

        cont = 0
        for i in range(len(recallMat)):
            if recallMat[i, col] >= eval_hparams['AREA_RECALL_CONSTRAINT'] and \
               precisionMat[i, col] >= eval_hparams['AREA_PRECISION_CONSTRAINT']:
                cont += 1
        if cont != 1:
            return False

        if recallMat[row, col] >= eval_hparams['AREA_RECALL_CONSTRAINT'] and \
           precisionMat[row, col] >= eval_hparams['AREA_PRECISION_CONSTRAINT']:
            return True
        return False

    def one_to_one_match_v2(row, col):
        """
        대체 방법을 사용하여 일대일 매칭을 확인합니다.

        Args:
            row (int): 실제 바운딩 박스의 인덱스.
            col (int): 예측 바운딩 박스의 인덱스.

        Returns:
            bool: 일대일 매칭 조건을 만족하면 True, 그렇지 않으면 False.
        """
        if row_sum[row] != 1:
            return False

        if col_sum[col] != 1:
            return False

        if recallMat[row, col] >= eval_hparams['AREA_RECALL_CONSTRAINT'] and \
           precisionMat[row, col] >= eval_hparams['AREA_PRECISION_CONSTRAINT']:
            return True
        return False

    def num_overlaps_gt(gtNum):
        """
        특정 실제 바운딩 박스(gtNum)와 겹치는 예측 바운딩 박스의 수를 셉니다.

        Args:
            gtNum (int): 실제 바운딩 박스의 인덱스.

        Returns:
            int: 겹치는 예측 바운딩 박스의 수.
        """
        cont = 0
        for detNum in range(len(detRects)):
            if recallMat[gtNum, detNum] > 0:
                cont += 1
        return cont

    def num_overlaps_det(detNum):
        """
        특정 예측 바운딩 박스(detNum)와 겹치는 실제 바운딩 박스의 수를 셉니다.

        Args:
            detNum (int): 예측 바운딩 박스의 인덱스.

        Returns:
            int: 겹치는 실제 바운딩 박스의 수.
        """
        cont = 0
        for gtNum in range(len(recallMat)):
            if recallMat[gtNum, detNum] > 0:
                cont += 1
        return cont

    def is_single_overlap(row, col):
        """
        실제 바운딩 박스(row)와 예측 바운딩 박스(col)가 각각 단 하나의 상호 겹침을 가지는지 확인합니다.

        Args:
            row (int): 실제 바운딩 박스의 인덱스.
            col (int): 예측 바운딩 박스의 인덱스.

        Returns:
            bool: 양쪽 모두 단일 겹침이면 True, 그렇지 않으면 False.
        """
        if num_overlaps_gt(row) == 1 and num_overlaps_det(col) == 1:
            return True
        else:
            return False

    def one_to_many_match(gtNum):
        """
        실제 바운딩 박스(gtNum)가 여러 예측 바운딩 박스와 매칭되는지 확인합니다.

        Args:
            gtNum (int): 실제 바운딩 박스의 인덱스.

        Returns:
            tuple:
                - bool: 일대다 매칭이 발견되면 True, 그렇지 않으면 False.
                - list: 매칭된 예측 바운딩 박스의 인덱스 리스트.
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
        예측 바운딩 박스(detNum)가 여러 실제 바운딩 박스와 매칭되는지 확인합니다.

        Args:
            detNum (int): 예측 바운딩 박스의 인덱스.

        Returns:
            tuple:
                - bool: 다대일 매칭이 발견되면 True, 그렇지 않으면 False.
                - list: 매칭된 실제 바운딩 박스의 인덱스 리스트.
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
        두 직사각형(a, b)의 교집합 면적을 계산합니다.

        Args:
            a (namedtuple): 직사각형 a, 속성: xmin, ymin, xmax, ymax.
            b (namedtuple): 직사각형 b, 속성: xmin, ymin, xmax, ymax.

        Returns:
            float: 교집합 면적. 겹치지 않으면 0.0을 반환.
        """
        dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin) + 1
        dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin) + 1
        if (dx >= 0) and (dy >= 0):
            return dx * dy
        else:
            return 0.0

    def center(r):
        """
        직사각형의 중심점을 계산합니다.

        Args:
            r (namedtuple): 직사각형, 속성: xmin, ymin, xmax, ymax.

        Returns:
            namedtuple: 중심점을 나타내는 Point, 속성: x, y.
        """
        x = float(r.xmin) + float(r.xmax - r.xmin + 1) / 2.0
        y = float(r.ymin) + float(r.ymax - r.ymin + 1) / 2.0
        return Point(x, y)

    def point_distance(r1, r2):
        """
        두 점(r1, r2) 사이의 유클리드 거리를 계산합니다.

        Args:
            r1 (namedtuple): 첫 번째 점, 속성: x, y.
            r2 (namedtuple): 두 번째 점, 속성: x, y.

        Returns:
            float: 두 점 사이의 거리.
        """
        distx = math.fabs(r1.x - r2.x)
        disty = math.fabs(r1.y - r2.y)
        return math.sqrt(distx * distx + disty * disty)

    def center_distance(r1, r2):
        """
        두 직사각형(r1, r2)의 중심점 사이의 거리를 계산합니다.

        Args:
            r1 (namedtuple): 첫 번째 직사각형.
            r2 (namedtuple): 두 번째 직사각형.

        Returns:
            float: 두 중심점 사이의 거리.
        """
        return point_distance(center(r1), center(r2))

    def diag(r):
        """
        직사각형의 대각선 길이를 계산합니다.

        Args:
            r (namedtuple): 직사각형, 속성: xmin, ymin, xmax, ymax.

        Returns:
            float: 직사각형의 대각선 길이.
        """
        w = (r.xmax - r.xmin + 1)
        h = (r.ymax - r.ymin + 1)
        return math.sqrt(h * h + w * w)

    # 평가 매개변수가 제공되지 않은 경우 기본 매개변수를 사용
    if eval_hparams is None:
        eval_hparams = default_evaluation_params()

    # 현재는 'rect' 형식만 지원
    if bbox_format != 'rect':
        raise NotImplementedError("현재 'rect' 형식의 바운딩 박스만 지원됩니다.")

    # 바운딩 박스가 직사각형 형식이 아닌 경우 직사각형으로 변환
    _pred_bboxes_dict, _gt_bboxes_dict = deepcopy(pred_bboxes_dict), deepcopy(gt_bboxes_dict)
    pred_bboxes_dict, gt_bboxes_dict = dict(), dict()

    # 예측 바운딩 박스 처리
    for sample_name, bboxes in _pred_bboxes_dict.items():
        # 이미 직사각형 형식인 경우 그대로 사용
        if len(bboxes) > 0 and np.array(bboxes[0]).ndim == 1 and len(bboxes[0]) == 4:
            pred_bboxes_dict = _pred_bboxes_dict
            break

        pred_bboxes_dict[sample_name] = []
        for bbox in map(np.array, bboxes):
            # [xmin, ymin, xmax, ymax] 형식으로 변환
            rect = [bbox[:, 0].min(), bbox[:, 1].min(), bbox[:, 0].max(), bbox[:, 1].max()]
            pred_bboxes_dict[sample_name].append(rect)

    # 실제 바운딩 박스 처리
    for sample_name, bboxes in _gt_bboxes_dict.items():
        # 이미 직사각형 형식인 경우 그대로 사용
        if len(bboxes) > 0 and np.array(bboxes[0]).ndim == 1 and len(bboxes[0]) == 4:
            gt_bboxes_dict = _gt_bboxes_dict
            break

        gt_bboxes_dict[sample_name] = []
        for bbox in map(np.array, bboxes):
            # [xmin, ymin, xmax, ymax] 형식으로 변환
            rect = [bbox[:, 0].min(), bbox[:, 1].min(), bbox[:, 0].max(), bbox[:, 1].max()]
            gt_bboxes_dict[sample_name].append(rect)

    # 메트릭스를 저장할 구조 초기화
    perSampleMetrics = {}

    methodRecallSum = 0
    methodPrecisionSum = 0

    # Rectangle과 Point를 위한 namedtuple 정의
    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    Point = namedtuple('Point', 'x y')

    numGt = 0
    numDet = 0

    # 실제 바운딩 박스 사전의 각 샘플에 대해 반복
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

        # 현재 샘플의 실제 바운딩 박스 가져오기
        pointsList = gt_bboxes_dict[sample_name]

        for n in range(len(pointsList)):
            points = pointsList[n]
            gtRect = Rectangle(*points)
            gtRects.append(gtRect)
            gtPolPoints.append(np.array(points).tolist())

        evaluationLog += "GT rectangles: " + str(len(gtRects)) + '\n'

        # 현재 샘플에 예측 바운딩 박스가 있는지 확인
        if sample_name in pred_bboxes_dict:
            pointsList = pred_bboxes_dict[sample_name]

            for n in range(len(pointsList)):
                points = pointsList[n]
                detRect = Rectangle(*points)
                detRects.append(detRect)
                detPolPoints.append(np.array(points).tolist())

            evaluationLog += "DET rectangles: " + str(len(detRects)) + '\n'

            if len(gtRects) == 0:
                # 실제 바운딩 박스가 없으면 재현율은 1, 정밀도는 예측이 있으면 0, 없으면 1
                recall = 1
                precision = 0 if len(detRects) > 0 else 1

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

                # 면적 제약 조건을 충족하는지 확인
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
                                # 중심 거리 기반 추가 검증
                                if is_single_overlap(gtNum, detNum) is True:
                                    rG = gtRects[gtNum]
                                    rD = detRects[detNum]
                                    normDist = center_distance(rG, rD)
                                    normDist /= diag(rG) + diag(rD)
                                    normDist *= 2.0
                                    if normDist < eval_hparams['EV_PARAM_IND_CENTER_DIFF_THR']:
                                        # 유효한 일대일 매칭
                                        gtRectMat[gtNum] = 1
                                        detRectMat[detNum] = 1
                                        recallAccum += eval_hparams['MTYPE_OO_O']
                                        precisionAccum += eval_hparams['MTYPE_OO_O']
                                        pairs.append({'gt': gtNum, 'det': detNum, 'type': 'OO'})
                                        evaluationLog += "Match GT #" + str(gtNum) + " with Det #" + str(detNum) + "\n"
                                    else:
                                        # 정규화된 거리 임계값 때문에 매칭 폐기
                                        evaluationLog += "Match Discarded GT #" + str(gtNum) + \
                                                          " with Det #" + str(detNum) + \
                                                          " normDist: " + str(normDist) + " \n"
                                else:
                                    # 다중 겹침 때문에 매칭 폐기
                                    evaluationLog += "Match Discarded GT #" + str(gtNum) + \
                                                      " with Det #" + str(detNum) + \
                                                      " not single overlap\n"

                # 일대다 매칭 찾기
                evaluationLog += "Find one-to-many matches\n"
                for gtNum in range(len(gtRects)):
                    match, matchesDet = one_to_many_match(gtNum)
                    if match is True:
                        evaluationLog += "num_overlaps_gt=" + str(num_overlaps_gt(gtNum))
                        # 겹침 수에 따른 추가 검증
                        if num_overlaps_gt(gtNum) >= 2:
                            gtRectMat[gtNum] = 1
                            if len(matchesDet) == 1:
                                recallAccum += eval_hparams['MTYPE_OO_O']
                                precisionAccum += eval_hparams['MTYPE_OO_O']
                            else:
                                recallAccum += eval_hparams['MTYPE_OM_O']
                                precisionAccum += eval_hparams['MTYPE_OM_O'] * len(matchesDet)
                            # 매칭된 쌍 기록
                            pairs.append({'gt': gtNum, 'det': matchesDet, 'type': 'OO' if len(matchesDet) == 1 else 'OM'})
                            for detNum in matchesDet:
                                detRectMat[detNum] = 1
                            evaluationLog += "Match GT #" + str(gtNum) + " with Det #" + str(matchesDet) + "\n"
                        else:
                            # 겹침 수 부족으로 매칭 폐기
                            evaluationLog += "Match Discarded GT #" + str(gtNum) + \
                                              " with Det #" + str(matchesDet) + \
                                              " not single overlap\n"

                # 다대일 매칭 찾기
                evaluationLog += "Find many-to-one matches\n"
                for detNum in range(len(detRects)):
                    match, matchesGt = many_to_one_match(detNum)
                    if match is True:
                        # 겹침 수에 따른 추가 검증
                        if num_overlaps_det(detNum) >= 2:
                            detRectMat[detNum] = 1
                            if len(matchesGt) == 1:
                                recallAccum += eval_hparams['MTYPE_OO_O']
                                precisionAccum += eval_hparams['MTYPE_OO_O']
                            else:
                                recallAccum += eval_hparams['MTYPE_OM_M'] * len(matchesGt)
                                precisionAccum += eval_hparams['MTYPE_OM_M']
                            # 매칭된 쌍 기록
                            pairs.append({'gt': matchesGt, 'det': detNum, 'type': 'OO' if len(matchesGt) == 1 else 'MO'})
                            for gtNum in matchesGt:
                                gtRectMat[gtNum] = 1
                            evaluationLog += "Match GT #" + str(matchesGt) + " with Det #" + str(detNum) + "\n"
                        else:
                            # 겹침 수 부족으로 매칭 폐기
                            evaluationLog += "Match Discarded GT #" + str(matchesGt) + \
                                              " with Det #" + str(detNum) + \
                                              " not single overlap\n"

                # 현재 샘플의 실제 바운딩 박스 수
                numGtCare = len(gtRects)
                if numGtCare == 0:
                    recall = float(1)
                    precision = float(0) if len(detRects) > 0 else float(1)
                else:
                    recall = float(recallAccum) / numGtCare
                    precision = float(0) if len(detRects) == 0 else float(precisionAccum) / len(detRects)
                hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)

        # 전체 재현율과 정밀도 누적
        methodRecallSum += recallAccum
        methodPrecisionSum += precisionAccum
        numGt += len(gtRects)
        numDet += len(detRects)

        # 현재 샘플의 메트릭스 저장
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

    # 전체 메트릭스 계산
    methodRecall = 0 if numGt == 0 else methodRecallSum / numGt
    methodPrecision = 0 if numDet == 0 else methodPrecisionSum / numDet
    methodHmean = 0 if (methodRecall + methodPrecision) == 0 else \
        2 * methodRecall * methodPrecision / (methodRecall + methodPrecision)

    # 메서드 수준의 메트릭스 정리
    methodMetrics = {'precision': methodPrecision, 'recall': methodRecall, 'hmean': methodHmean}

    # 최종 결과 사전 준비
    resDict = {
        'calculated': True,
        'Message': '',
        'total': methodMetrics,
        'per_sample': perSampleMetrics,
        'eval_hparams': eval_hparams
    }

    return resDict


# # 추가 유틸리티 클래스

# class Rectangle(namedtuple('Rectangle', 'xmin ymin xmax ymax')):
#     """
#     좌표를 가진 직사각형을 나타내는 namedtuple.

#     Attributes:
#         xmin (float): 최소 x-좌표 (왼쪽).
#         ymin (float): 최소 y-좌표 (위쪽).
#         xmax (float): 최대 x-좌표 (오른쪽).
#         ymax (float): 최대 y-좌표 (아래쪽).
#     """
#     pass


# class Point(namedtuple('Point', 'x y')):
#     """
#     2D 공간의 점을 나타내는 namedtuple.

#     Attributes:
#         x (float): x-좌표.
#         y (float): y-좌표.
#     """
#     pass
