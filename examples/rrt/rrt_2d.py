# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
import numpy as np

from src.rrt.rrt import RRT
from src.search_space.search_space import SearchSpace
from src.utilities.plotting import Plot

# 전체 맵의 크기
X_dimensions = np.array([(0, 100), (0, 100)])  
# 장애물의 위치
Obstacles = np.array([(20, 20, 40, 40), (20, 60, 40, 80), (60, 20, 80, 40), (60, 60, 80, 80)])
# 출발점
x_init = (0, 0)  
# 도착점
x_goal = (100, 100)  

# 한 step의 길이
Q = np.array([(8, 4)])  
# 장애물과의 교점을 판단하기 위한 가장 작은 edges의 길이
r = 1 
# 시간 초과 전에 취할 최대 샘플 수
max_samples = 1024
# 목적지와의 연결을 확인할 확률
prc = 0.1  

# 전체 맵의 크기와 장애물의 위치들을 클래스인 SearchSpace에 보낸다.
X = SearchSpace(X_dimensions, Obstacles)

# create rrt_search
# src.rrt.rrt.py에 있는 RRT로 호출
# X를 보냈다는 것은 SearchSpace에 접근할 수 있도록 해주는 것이다.
rrt = RRT(X, Q, x_init, x_goal, max_samples, r, prc)
path = rrt.rrt_search()   # 여기에서 search_space.py를 부르게 된다.

# plot
plot = Plot("rrt_2d")
plot.plot_tree(X, rrt.trees)
if path is not None:
    plot.plot_path(X, path)
plot.plot_obstacles(X, Obstacles)
plot.plot_start(X, x_init)
plot.plot_goal(X, x_goal)
plot.draw(auto_open=True)

'''
코드의 전체 적인 구조
rrt_2d.py --> rrt.py --> rrt_base.py(잡다한 일을 제일 많이 한다.)
src를 안으로 넣기
numpy 버전 낮추기
html 파일 가져오기(깃허브)
plotting.py 제일 마지막줄 바꾸기
'''