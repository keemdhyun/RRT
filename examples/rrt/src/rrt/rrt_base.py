import random

import numpy as np

from src.rrt.tree import Tree
from src.utilities.geometry import steer

# rrt.py에서 불리자마자 제일 처음을오 호출
class RRTBase(object):
    def __init__(self, X, Q, x_init, x_goal, max_samples, r, prc=0.01):
        """
        Template RRT planner
        :param X: Search Space
        :param Q: list of lengths of edges added to tree
        :param x_init: tuple, initial location
        :param x_goal: tuple, goal location
        :param max_samples: max number of samples to take
        :param r: resolution of points to sample along edge when checking for collisions
        :param prc: probability of checking whether there is a solution
        """
        self.X = X
        self.samples_taken = 0
        self.max_samples = max_samples
        self.Q = Q
        self.r = r
        self.prc = prc
        self.x_init = x_init
        self.x_goal = x_goal
        self.trees = []  # list of all trees
        # def add_tree를 호출
        self.add_tree()  # add initial tree
        self.count = 0

    # 함수 발동 : RRTBase의 생성자에서 호출
    # 입력 변수 : 없다.
    # 함수 설명
    # 반환 변수 : 없다.
    def add_tree(self):
        """
        Create an empty tree and add to trees
        """
        # src.rrt.tree.py의 Tree를 호출
        self.trees.append(Tree(self.X))

    # 함수 발동 : rrt.py의 def rrt_search에서 호출, connect_to_point에서 호출
    # 입력 변수 : 0(tree), (0, 0)(v)
    # 반환 변수 : 없다.
    def add_vertex(self, tree, v):
        """
        트리에 정점을 추가
        :param tree: 정점을 추가할 트리
        :param v: 튜플 형태의 추가할 정점
        """
        # 정점을 추가하는 과정인듯
        self.trees[tree].V.insert(0, v + v, v)
        # 트리에 정점하나 카운트 추가
        self.trees[tree].V_count += 1
        # 샘플 카운트도 추가
        self.samples_taken += 1  

    # 함수 발동 : def connect_to_point에서 호출
    # 입력 변수 : 0(tree), 내분점(child), 가장 가까운 점(parent)
    # 반환 변수 : 없다.
    def add_edge(self, tree, child, parent):
        # child에 parent의 정보를 넣는다.(역추적할 수 있게끔)
        print(parent)
        self.trees[tree].E[child] = parent

    # 함수 발동 : get_nearest에서 호출
    # 입력 변수 : 0(tree), x_rand(x), 1(n)
    # 반환 변수 : x_new, x_nearest
    def nearby(self, tree, x, n):
        """
        :param tree: 검색 중인 트리
        :param x: 검색할 정점
        :param n: 반환할 최대 이웃수
        """
        # .nearest는 from rtree import index(내장된 모듈)에서 있는 함수인 듯    https://rtree.readthedocs.io/en/latest/tutorial.html
        return self.trees[tree].V.nearest(x, num_results=n, objects="raw")

    # 함수 발동 : new_and_near에서 호출,     can_connect_to_goal에서 호출
    # 입력 변수 : 0(tree), x_rand(x)         0(tree), self.x_goal(x)
    # 반환 변수 : x_nearest
    def get_nearest(self, tree, x):
        """
        :param tree: 검색 중인 트리
        :param x: 검색할 정점
        :return: x에 가장 가까운 정점
        """
        # next는 파이썬에 내장된 함수이다. 값을 빼내는 역할을 한다. https://homzzang.com/b/py-136
        return next(self.nearby(tree, x, 1))

    # 함수 발동rrt.py에 있는 rrt_search에서 호출
    # 입력 변수 : 0(tree), [8 4](q)
    # 반환 변수 : 새로 생성된 내분점(x_new), 가장 가까운 점(x_nearest)
    def new_and_near(self, tree, q):
        """
        Return a new steered vertex and the vertex in tree that is nearest
        :param tree: int, tree being searched
        :param q: length of edge when steering
        :return: vertex, new steered vertex, vertex, nearest vertex in tree to new vertex
        """
        # search_space.py에 있는 def sample_free 호출해서 랜덤인 좌표값 얻어낸다.
        x_rand = self.X.sample_free()
        # def get_nearest를 호출 x_rand와 가장 가까운 정점을 받아온다.
        x_nearest = self.get_nearest(tree, x_rand)
        # q[0] = 8
        # steer 파이썬 내장함수는 방향을 유지한 채로 내분점을 만들어주는 내장 함수다. (시점, 종점, 길이)
        # bound_point로 보내서 경계 안에 있도록 만들어준다.
        x_new = self.bound_point(steer(x_nearest, x_rand, q[0]))
        # x_new 포인트가 V안에 혼자만 존재하는지 체크(.count)(중복되지 않는지), x_new가 장애물의 위치에 있지는 않은지
        if not self.trees[0].V.count(x_new) == 0 or not self.X.obstacle_free(x_new):
            return None, None
        # 샘플을 하나씩 카운트해준다.
        self.samples_taken += 1
        return x_new, x_nearest

    # 함수 발동 : rrt.py에 있는 rrt_search에서 호출
    # 입력 변수 : 0(tree), 가장 가까운 점(x_a), 새로 생성된 내분점(x_b)
    # 함수 설명 : 새로 생성된 내분점이 여태껏 없던 것인가 질문 and 장애물과는 충돌하지 않는지 체크
    # 반환 변수 : 새로 생성된 내분점(x_new), 가장 가까운 점(x_nearest)
    def connect_to_point(self, tree, x_a, x_b):
        """
        :return: bool, True if able to add edge, False if prohibited by an obstacle
        """
        if self.trees[tree].V.count(x_b) == 0 and self.X.collision_free(x_a, x_b, self.r):
            self.add_vertex(tree, x_b)
            self.add_edge(tree, x_b, x_a)
            return True
        return False

    # get_path에서 호출
    # 입력 변수 : 0(tree)
    # 함수 설명 : 목표가 그래프에 연결될 수 있는지 확인
    # 반환 변수 : 가능한 경우 true 아닌 경우 false
    def can_connect_to_goal(self, tree):
        x_nearest = self.get_nearest(tree, self.x_goal)
        # 트리 안에 목표가 있는 것과 트리 한 칸 위에 가장 가까운 정점이 있는지를 체크
        if self.x_goal in self.trees[tree].E and x_nearest in self.trees[tree].E[self.x_goal]:
            return True
        # 마지막에 직선의 형태가 나왔던 이유
        # 목표로부터 가장 가까운 점과 직선으로 이었을 때 장애물과 부딪히지 않았을 때
        # 직진
        if self.X.collision_free(x_nearest, self.x_goal, self.r):  # check if obstacle-free
            return True
        return False

    # check_solution에서 호출
    # 입력 변수 : 없다.
    # 함수 설명 : 시점에 종점까지 트리를 통한 반환 경로
    # 반환 변수 : 가능한 경우 경로를 반환하고 그렇지 않은 경우에 반환하지 않는다.
    def get_path(self):
        """
        Return path through tree from start to goal
        :return: path if possible, None otherwise
        """
        if self.can_connect_to_goal(0):
            print("Can connect to goal")
            self.connect_to_goal(0)
            return self.reconstruct_path(0, self.x_init, self.x_goal)
        print("Could not connect to goal")
        return None

    # get_path 마지막 조금 위에서 호출
    # 입력 변수 : 0(tree)
    # 함수 설명 : 목적지 즉 종점과 그래프를 연결한다.
    # 반환 변수 : 없다.
    def connect_to_goal(self, tree):
        """
        Connect x_goal to graph
        """
        x_nearest = self.get_nearest(tree, self.x_goal)
        self.trees[tree].E[self.x_goal] = x_nearest

    # get_path 마지막에서 호출
    # 입력 변수 : 0(tree), 출발점(x_init), 도착점(x_goal)
    # 함수 설명 : 역추적해서 쭉 찾아간다.
    # 반환 변수 : 경로
    def reconstruct_path(self, tree, x_init, x_goal):
        """
        Reconstruct path from start to goal
        :return: sequence of vertices from start to goal
        """
        path = [x_goal]
        current = x_goal
        # 예외 체크
        if x_init == x_goal:
            return path
        
        # 부모가 출발점이 나올 때까지 경로를 역순으로 생성
        while not self.trees[tree].E[current] == x_init:
            path.append(self.trees[tree].E[current])
            # 부모 업데이트
            current = self.trees[tree].E[current]
        # 출발점도 추가
        path.append(x_init)
        # 역추적했으니 다시 역순으로 정렬
        path.reverse()
        return path

    # 함수 발동 : rrt.py에 있는 rrt_search에서 호출
    # 입력 변수 : 없다.
    # 함수 설명 : 
    # 반환 변수 : 
    def check_solution(self):
        # 해결책을 찾았는지 확률적으로 확인
        # random.random() random모듈의 random()함수를 호출하면
        # 0이상 1미만의 숫자 중 아무 숫자를 돌려줌
        if self.prc and random.random() < self.prc:
            print("Checking if can connect to goal at", str(self.samples_taken), "samples")
            path = self.get_path()
            if path is not None:
                return True, path
        # 내가 설정한 max_samples에 도달한 경우 그냥 반환해버리기
        if self.samples_taken >= self.max_samples:
            return True, self.get_path()
        return False, None

    # 함수 발동 : new_and_near에서 호출한다.
    # 입력 변수 : 방향을 유지한 내분점
    # 함수 설명 : 포인트가 경계를 벗어나면 경계로 설정
    # 반환 변수 : 방향을 유지한 내분점
    def bound_point(self, point):
        # np.maximum함수 : 두 좌표를 비교해서 더 큰 x, y좌표를 가져온다. https://jimmy-ai.tistory.com/70
        # np.minimum함수는 반대
        point = np.maximum(point, self.X.dimension_lengths[:, 0])
        point = np.minimum(point, self.X.dimension_lengths[:, 1])
        return tuple(point)