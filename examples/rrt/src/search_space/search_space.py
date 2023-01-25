# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

import numpy as np
from rtree import index

from src.utilities.geometry import es_points_along_line
from src.utilities.obstacle_generation import obstacle_generator

# 함수 발동 : rrt_2d.py에서 호출
# 입력 변수 : 맵의 크기 dimension_lengths, 장애물 들의 위치
# 함수 내용
# 반환 변수 : self.obs.count(x)가 0인지 아닌지를 체크
class SearchSpace(object):
    def __init__(self, dimension_lengths, O=None):
        """
        Initialize Search Space
        :param dimension_lengths: range of each dimension
        :param O: list of obstacles
        """
        # sanity check
        if len(dimension_lengths) < 2:
            raise Exception("Must have at least 2 dimensions")
        self.dimensions = len(dimension_lengths)  # number of dimensions
        # sanity checks
        if any(len(i) != 2 for i in dimension_lengths):
            raise Exception("Dimensions can only have a start and end")
        if any(i[0] >= i[1] for i in dimension_lengths):
            raise Exception("Dimension start must be less than dimension end")
        self.dimension_lengths = dimension_lengths  # length of each dimension
        p = index.Property()
        p.dimension = self.dimensions
        if O is None:
            self.obs = index.Index(interleaved=True, properties=p)
        else:
            # r-tree representation of obstacles
            # sanity check
            if any(len(o) / 2 != len(dimension_lengths) for o in O):
                raise Exception("Obstacle has incorrect dimension definition")
            if any(o[i] >= o[int(i + len(o) / 2)] for o in O for i in range(int(len(o) / 2))):
                raise Exception("Obstacle start must be less than obstacle end")
            # src.utilities.obstacle_generation.py에 있는 obstacle_generator를 호출한다.
            self.obs = index.Index(obstacle_generator(O), interleaved=True, properties=p)

    # 여기 아래서부터는 나중에 체크할 때 계속 호출된다.

    # 함수 발동 : def sample_free에서 호출한다, rrt_base.py에 있는 new_and_near에서 호출(새로 생성한 내분점이 장애물에 위치하는가 체크하기 위해서)
    # 입력 변수
    # 함수 내용 
    # 위치가 장애물 내에 있는지 확인하는 함수 , x = 확인할 위치
    # 반환 변수 : self.obs.count(x)가 0인지 아닌지를 체크 장애물 안에 있지 않으면 true
    # 그렇지 않으면 false
    def obstacle_free(self, x):
        return self.obs.count(x) == 0

    # 함수 발동 : rrt_base.py에 있는 new_and_near에서 호출한다.
    # 입력 변수 : 없다.
    # 함수 내용
    # if self.obstacle_free(x): 장애물 내부가 아닐 때까지 계속해서 샘플링
    # 반환 변수 : self.obs.count(x)가 0인지 아닌지를 체크
    def sample_free(self):
        """
        Sample a location within X_free
        :return: random location within X_free
        """
        while True:  
            x = self.sample()
            if self.obstacle_free(x):
                return x

    # 함수 발동 : 에서 호출한다.
    # 입력 변수
    # 함수 내용
    # 반환 변수 : self.obs.count(x)가 0인지 아닌지를 체크
    def collision_free(self, start, end, r):
        """
        Check if a line segment intersects an obstacle
        :param start: starting point of line
        :param end: ending point of line
        :param r: resolution of points to sample along edge when checking for collisions
        :return: True if line segment does not intersect an obstacle, False otherwise
        """
        # src.utilities.geometry.py 안에 들어있는 es_points_along_line에 보낸다.
        points = es_points_along_line(start, end, r)
        coll_free = all(map(self.obstacle_free, points))
        return coll_free

    # 함수 발동 : def sample_free에서 호출한다.
    # 입력 변수
    # 함수 내용 
    # [[  0 100!]
    # [  0 100@]]
    # print(self.dimension_lengths)
    # print(self.dimension_lengths[0,:]) # []하나 안으로 들어가서 0번째 [] 여기선 [0 100!]
    # print(self.dimension_lengths[1,:]) # []하나 안으로 들어가서 0번째 [] 여기선 [0 100@]
    # print(self.dimension_lengths[:, 0]) # []하나 안으로 들어가서 0번째 [] 여기선 [0 0]
    # print(self.dimension_lengths[:, 1]) # []하나 안으로 들어가서 0번째 [] 여기선 [100! 100@]
    # np.random.uniform(low, high, size)
    # 반환 변수 : tuple(x)   ex (74.12, 12.66)
    def sample(self):
        x = np.random.uniform(self.dimension_lengths[:, 0], self.dimension_lengths[:, 1])
        
        # print(x)        # [,]
        # print(tuple(x)) # (,)

        return tuple(x)
