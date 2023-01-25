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

    # 함수 발동 : rrt.py의 def rrt_search에서 호출 connect_to_point에서 호출
    # 입력 변수 : 0(tree), (0, 0)(v)
    # 반환 변수 : 없다.
    def add_vertex(self, tree, v):
        """
        트리에 정점을 추가
        :param tree: 정점을 추가할 트리
        :param v: 튜플 형태의 추가할 정점
        """
        self.trees[tree].V.insert(0, v + v, v)
        # 트리에 정점하나 카운트 추가
        self.trees[tree].V_count += 1
        # 샘플 카운트도 추가
        self.samples_taken += 1  

    def add_edge(self, tree, child, parent):
        """
        Add edge to corresponding tree
        :param tree: int, tree to which to add vertex
        :param child: tuple, child vertex
        :param parent: tuple, parent vertex
        """
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

    # 함수 발동 : new_and_near에서 호출
    # 입력 변수 : 0(tree), x_rand(x)
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
    # 입력 변수 : 0(tree), (x_a), (x_b)
    # 반환 변수 : 새로 생성된 내분점(x_new), 가장 가까운 점(x_nearest)
    def connect_to_point(self, tree, x_a, x_b):
        """
        Connect vertex x_a in tree to vertex x_b
        :param tree: int, tree to which to add edge
        :param x_a: tuple, vertex
        :param x_b: tuple, vertex
        :return: bool, True if able to add edge, False if prohibited by an obstacle
        """
        if self.trees[tree].V.count(x_b) == 0 and self.X.collision_free(x_a, x_b, self.r):
            self.add_vertex(tree, x_b)
            self.add_edge(tree, x_b, x_a)
            return True
        return False

    def can_connect_to_goal(self, tree):
        """
        Check if the goal can be connected to the graph
        :param tree: rtree of all Vertices
        :return: True if can be added, False otherwise
        """
        x_nearest = self.get_nearest(tree, self.x_goal)
        if self.x_goal in self.trees[tree].E and x_nearest in self.trees[tree].E[self.x_goal]:
            # tree is already connected to goal using nearest vertex
            return True
        if self.X.collision_free(x_nearest, self.x_goal, self.r):  # check if obstacle-free
            return True
        return False

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

    def connect_to_goal(self, tree):
        """
        Connect x_goal to graph
        (does not check if this should be possible, for that use: can_connect_to_goal)
        :param tree: rtree of all Vertices
        """
        x_nearest = self.get_nearest(tree, self.x_goal)
        self.trees[tree].E[self.x_goal] = x_nearest

    def reconstruct_path(self, tree, x_init, x_goal):
        """
        Reconstruct path from start to goal
        :param tree: int, tree in which to find path
        :param x_init: tuple, starting vertex
        :param x_goal: tuple, ending vertex
        :return: sequence of vertices from start to goal
        """
        path = [x_goal]
        current = x_goal
        if x_init == x_goal:
            return path
        while not self.trees[tree].E[current] == x_init:
            path.append(self.trees[tree].E[current])
            current = self.trees[tree].E[current]
        path.append(x_init)
        path.reverse()
        return path

    def check_solution(self):
        # probabilistically check if solution found
        if self.prc and random.random() < self.prc:
            print("Checking if can connect to goal at", str(self.samples_taken), "samples")
            path = self.get_path()
            if path is not None:
                return True, path
        # check if can connect to goal after generating max_samples
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