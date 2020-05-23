import sys
import os
from IPython.display import HTML
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt

# from work1 import *
from work02 import SarsaModel, ValueIterationModel
from work01 import MazeModel, SimpleModel, PolicyGradientMethodModel
from logzero import logging, logger
import logzero


np.random.seed(8)


class QLearningModel(ValueIterationModel):
    # FIXME:使わない引数どうしよう
    def update_Q(self, state, action, reward, state_next, action_next):
        if state_next == 8:
            return self.Q[state, action] + self.ETA * (reward - self.Q[state, action])
        else:
            return self.Q[state, action] + self.ETA * (
                reward
                + self.GAMMA * np.nanmax(self.Q[state_next, :])
                - self.Q[state, action]
            )


# [a, b] = theta_0.shape
# Q = np.random.rand(a, b) * theta_0 * 0.1

# eta = 0.1
# gamma = 0.9
# epsilon = 0.5
# v = np.nanmax(Q, axis=1)
# is_continue = True
# episode = 1
# V = []
# V.append(np.nanmax(Q, axis=1))
# goal_maze_ret_s_a_Q=SarsaModel.goal_maze_ret_s_a_Q()

# while is_continue:
#     print("episode:" + str(episode))
#     episode = epsilon / 2
#     [s_a_history, Q] = goal_maze_ret_s_a_Q(Q, epsilon, eta, gamma,pi_0)
#     new_v = np.nanmax(Q, axis=1)
#     print(np.sum(np.abs(new_v - v)))
#     v = new_v
#     v.append(v)
#     print("steps" + str(len(s_a_history) - 1))

#     episode = episode + 1
#     if episode > 100:
#         break


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
    elif model_no == 3:
        model = QLearningModel(
            theta_0, n_iter=n_iter, verbose=verbose, loglevel=loglevel
        )

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
    loglevel = log_dict["error"]
    main(model, n_iter, is_plot, verbose, loglevel)

    #######################################
