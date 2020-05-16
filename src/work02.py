import sys
import os
from IPython.display import HTML
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt

from work01 import MazeModel, SimpleModel, PolicyGradientMethodModel
from logzero import logging, logger
import logzero


np.random.seed(8)


class SarsaModel(MazeModel):
    def __init__(self, theta, n_iter, verbose, loglevel):
        super().__init__(theta, n_iter, verbose, loglevel)
        self.ETA = 0.1
        self.GAMMA = 0.9
        self.EPSILON = 0.5
        self.Q = None
        self.V = None

    def run(self):
        self.PI = self.simple_convert_into_pi_from_theta()
        self.Q = np.random.rand(self.THETA.shape[0], self.THETA.shape[1]) * self.THETA
        self.V = np.nanmax(self.Q, axis=1)
        logger.debug(self.V.shape)

        for itr in range(self.n_iter):
            logger.info(f"itr:{itr}")
            self.EPSILON /= 2
            s_a_history = self.goal_maze_ret_s_a_Q()
            new_v = np.nanmax(self.Q, axis=1)
            logger.debug(f"V :{np.sum(np.abs(new_v - self.V))}")
            self.V = new_v
            logging.info(f"# of step to goal:{len(s_a_history) - 1}.")
        return s_a_history

    def simple_convert_into_pi_from_theta(self):
        """割合"""
        [m, n] = self.THETA.shape
        pi = np.zeros((m, n))
        for i in range(0, m):
            pi[i, :] = self.THETA[i, :] / np.nansum(
                self.THETA[i, :]
            )  # np.nanを0としてsumしてくれる関数

        pi = np.nan_to_num(pi)  # np.nan -> 0

        return pi

    def goal_maze_ret_s_a_Q(self):
        state = 0
        action = action_next = self.get_action(state)
        s_a_history = [[0, np.nan]]

        while 1:
            action = action_next
            s_a_history[-1][1] = action
            state_next = self.get_s_next(state, action)
            s_a_history.append([state_next, np.nan])
            if state_next == 8:
                reward = 1
                action_next = np.nan
            else:
                reward = 0
                action_next = self.get_action(state_next)

            self.Q[state, action] = self.sarsa(
                state, action, reward, state_next, action_next
            )

            if state_next == 8:
                break
            else:
                state = state_next

        return s_a_history

    def get_action(self, state):
        if np.random.rand() < self.EPSILON:
            next_direction = np.random.choice(self.direction, p=self.PI[state, :])
        else:
            next_direction = self.direction[np.nanargmax(self.Q[state, :])]

        return self.direction_map[next_direction]

    def get_s_next(self, state, action):

        if action == self.direction_map["up"]:
            s_next = state - 3
        elif action == self.direction_map["right"]:
            s_next = state + 1
        elif action == self.direction_map["down"]:
            s_next = state + 3
        elif action == self.direction_map["left"]:
            s_next = state - 1

        return s_next

    def sarsa(self, state, action, reward, s_next, a_next):
        if s_next == 8:
            return self.Q[state, action] + self.ETA * (reward - self.Q[state, action])
        else:
            return self.Q[state, action] + self.ETA * (
                reward + self.GAMMA * self.Q[s_next, a_next] - self.Q[state, action]
            )


def main(model_no, n_iter, is_plot=False, verbose=False, loglevel=logging.ERROR):

    theta_0 = np.array(
        [
            [np.nan, 1, 1, np.nan],  # s0
            [np.nan, 1, np.nan, 1],  # s1
            [np.nan, np.nan, 1, 1],  # s2
            [1, 1, 1, np.nan],  # s3
            [np.nan, np.nan, 1, 1],  # s4
            [1, np.nan, np.nan, np.nan],  # s5
            [1, np.nan, np.nan, np.nan],  # s6
            [1, 1, np.nan, np.nan],  # s7
        ]
    )

    if model_no == 0:
        model = SimpleModel(theta_0)
    elif model_no == 1:
        model = PolicyGradientMethodModel(
            theta_0, n_iter=n_iter, verbose=verbose, loglevel=loglevel
        )
    elif model_no == 2:
        model = SarsaModel(theta_0, n_iter=n_iter, verbose=verbose, loglevel=loglevel)

    # 実行
    logger.debug("start: run")
    s_a_history = model.run()
    logger.debug("end: run")

    print(s_a_history)
    print(f"# of step :{len(s_a_history)-1}")

    if is_plot:

        def init():
            line.set_data([], [])
            return (line,)

        def animate(i):
            if model_no == 0:
                state = s_a_history[i]
            elif model_no == 1:
                state = s_a_history[i][0]

            x = (state % 3) + 0.5
            y = 2.5 - int(state / 3)
            line.set_data(x, y)
            return (line,)

        anim = animation.FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=len(s_a_history),
            interval=299,
            repeat=False,
        )

        # HTML(anim.to_jshtml())
        plt.show()

        # show maze
        plt.savefig("output.png")


if __name__ == "__main__":
    log_dict = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    args = sys.argv

    is_plot = False
    verbose = False
    model = int(args[1])
    n_iter = int(args[2])
    loglevel = log_dict["debug"]
    main(model, n_iter, is_plot, verbose, loglevel)

    #######################################
