from src.rrt.rrt_base import RRTBase


# rrt_2d.py에서 RRT에 호출이 된 후에 바로 src.rrt.rrt_base.py의 RRTBase로 보내준다.
class RRT(RRTBase):
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
        super().__init__(X, Q, x_init, x_goal, max_samples, r, prc)

    # rrt_2d.py에서 호출한다.
    def rrt_search(self):
        """
        Create and return a Rapidly-exploring Random Tree, keeps expanding until can connect to goal
        https://en.wikipedia.org/wiki/Rapidly-exploring_random_tree
        :return: list representation of path, dict representing edges of tree in form E[child] = parent
        """

        # rrt_base.py에 있는 def add_vertex로 호출
        self.add_vertex(0, self.x_init)
        # rrt_base.py에 있는 def add_edge로 호출
        self.add_edge(0, self.x_init, None)

        # self.Q = rrt_2d.py에 있는 Q와 같다.

        while True:
            for q in self.Q:  # iterate over different edge lengths until solution found or time out
                for i in range(q[1]):  # iterate over number of edges of given length to add
                    # rrt_base.py에 있는 def new_and_near로 호출
                    # 0, q = [8 4]
                    # 새로 생성한 내분점, 가장 가까운 점을 획득
                    x_new, x_nearest = self.new_and_near(0, q)

                    # 오류 방지
                    if x_new is None:
                        continue

                    # 정점들 연결(기존의 가장 가까운 점과 새로 생성한 내분점)
                    # 여러 조건들을 만족하게끔해서 결국 선을 이어버린다.
                    self.connect_to_point(0, x_nearest, x_new)

                    solution = self.check_solution()
                    if solution[0]:
                        # 찾은 경로를 반환해준다.
                        return solution[1]
                    
        
