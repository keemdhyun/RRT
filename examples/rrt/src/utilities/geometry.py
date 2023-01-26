# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

from itertools import tee

import numpy as np

# 함수 발동 : rrt_base.py에서의 def connect_to_point에서 호출한다.
# 입력 변수 : 가장 가까운 점(a), 새로 생성된 내분점(b)
# 함수 내용
# 반환 변수 : a, b사이의 유클리드 거리를 반환
def dist_between_points(a, b):
    # https://www.delftstack.com/ko/howto/numpy/calculate-euclidean-distance/
    # 가끔은 이해할 필요가 없는 경우도 있다.
    distance = np.linalg.norm(np.array(b) - np.array(a))
    return distance

def pairwise(iterable):
    """
    Pairwise iteration over iterable
    :param iterable: iterable
    :return: s -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

# 함수 발동 : rrt_base.py에서의 def connect_to_point에서 호출한다.
# 입력 변수 : 가장 가까운 점(start), 새로 생성된 내분점(end), r(rrt_2d.py의 r)
# 함수 내용
# 반환 변수 : 거리 r로 구분된 시작부터 끝까지 선을 따라 점을 생성합니다.
def es_points_along_line(start, end, r):
    d = dist_between_points(start, end)
    # numpy.ceil : 주어진 숫자보다 큰 수 중에 가장 작은 정수
    n_points = int(np.ceil(d / r))
    if n_points > 1:
        step = d / (n_points - 1)
        for i in range(n_points):
            next_point = steer(start, end, i * step)
            yield next_point

# 함수 발동 : geometry.py에서의 es_points_along_line에서 호출한다.
# 입력 변수 : 가장 가까운 점(start), 새로 생성된 내분점(end), 떨구고 싶은 거리(d)
# 함수 내용
# 반환 변수 : 내분점을 생성해준다.
def steer(start, goal, d):
    """
    Return a point in the direction of the goal, that is distance away from start
    :return: point in the direction of the goal, distance away from start
    """
    start, end = np.array(start), np.array(goal)
    v = end - start
    # u = 단위 벡터
    u = v / (np.sqrt(np.sum(v ** 2)))
    steered_point = start + u * d
    return tuple(steered_point)
