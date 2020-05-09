import sys
import os
from IPython.display import HTML
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(8)


class MazeModel:
    def __init__(self, theta, n_iter=1, verbose=False):
        self.THETA = theta
        self.PI = None
        self.n_iter = n_iter
        self.verbose = verbose


class SimpleModel(MazeModel):
    def run(self):
        self.PI = self.simple_convert_into_pi_from_theta()
        return self.goal_maze(self.PI)

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

    # 1step 移動後の状態s
    def get_next_s(self, pi: np, s: int) -> int:

        direction = ["up", "right", "down", "left"]
        next_direction = np.random.choice(direction, p=pi[s, :])

        if next_direction == "up":
            s_next = s - 3
        elif next_direction == "right":
            s_next = s + 1
        elif next_direction == "down":
            s_next = s + 3
        elif next_direction == "left":
            s_next = s - 1

        return s_next

    def goal_maze(self, pi):
        s = 0
        state_history = [0]

        while 1:
            next_s = self.get_next_s(pi, s)
            state_history.append(next_s)

            if next_s == 8:
                break
            else:
                s = next_s

        return state_history


class PolicyGradientMethodModel(MazeModel):
    def run(self):
        if verbose:
            print(self.THETA)

        for iter in range(self.n_iter):
            self.PI = self.softmax_convert_into_pi_from_theta()
            s_a_history = self.goal_maze_ret_s_a(self.PI)
            self.THETA = self.update_theta(self.THETA, self.PI, s_a_history)
            print(f"# of step :{len(s_a_history)-1}")
            if verbose:
                print("-" * 5)

        print("↑,→,↓, ←")
        print(self.THETA)
        return s_a_history

    def softmax_convert_into_pi_from_theta(self):
        """
        softmax関数を用いて方策を求める
        """

        beta = 1.0
        [m, n] = self.THETA.shape
        pi = np.zeros((m, n))
        exp_theta = np.exp(beta * self.THETA)

        for i in range(0, m):
            pi[i, :] = exp_theta[i, :] / np.nansum(exp_theta[i, :])

        pi = np.nan_to_num(pi)

        return pi

    def get_action_and_next_s(self, pi: np, s: int) -> int:
        """
        方策と状態を入れると次の状態を返す
        """
        direction = ["up", "right", "down", "left"]
        next_direction = np.random.choice(direction, p=pi[s, :])

        if next_direction == "up":
            action = 0
            s_next = s - 3
        elif next_direction == "right":
            action = 1
            s_next = s + 1
        elif next_direction == "down":
            action = 2
            s_next = s + 3
        elif next_direction == "left":
            action = 3
            s_next = s - 1

        return [action, s_next]

    def goal_maze_ret_s_a(self, pi):
        s = 0
        s_a_history = [[0, np.nan]]

        while 1:
            [action, next_s] = self.get_action_and_next_s(pi, s)
            s_a_history[-1][1] = action
            s_a_history.append([next_s, np.nan])

            if next_s == 8:
                break
            else:
                s = next_s

        return s_a_history

    def update_theta(self, theta, pi, s_a_history):
        eta = 0.1
        T = len(s_a_history)

        [m, n] = theta.shape
        delta_theta = theta.copy()

        hist_array = np.zeros(self.THETA.shape)
        for [state, action] in s_a_history[:-1]:

            hist_array[state, action] += 1

        delta_theta = (
            hist_array - pi * np.sum(hist_array, axis=1).reshape((-1, 1))
        ) / T

        return theta + eta * delta_theta

        # for i in range(0, m):
        #     for j in range(0, n):
        #         if not np.isnan(theta[i, j]):
        # SA_i = [SA for SA in s_a_history if SA[0] == i]
        # SA_ij = [SA for SA in s_a_history if SA == [i, j]]

        # N_i = len(SA_i)
        # N_ij = len(SA_ij)
        # delta_theta[i, j] = (N_ij - pi[i, j] * N_i) / T


def create_maze():

    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    plt.plot([1, 1], [0, 1], color="red", linewidth=2)
    plt.plot([1, 2], [2, 2], color="red", linewidth=2)
    plt.plot([2, 2], [2, 1], color="red", linewidth=2)
    plt.plot([2, 3], [1, 1], color="red", linewidth=2)

    # 状態
    plt.text(0.5, 2.5, "S0", size=14, ha="center")
    plt.text(1.5, 2.5, "S1", size=14, ha="center")
    plt.text(2.5, 2.5, "S2", size=14, ha="center")
    plt.text(0.5, 1.5, "S3", size=14, ha="center")
    plt.text(1.5, 1.5, "S4", size=14, ha="center")
    plt.text(2.5, 1.5, "S5", size=14, ha="center")
    plt.text(0.5, 0.5, "S6", size=14, ha="center")
    plt.text(1.5, 0.5, "S7", size=14, ha="center")
    plt.text(2.5, 0.5, "S8", size=14, ha="center")
    plt.text(0.5, 2.3, "START", ha="center")
    plt.text(2.5, 0.3, "GOAL", ha="center")

    # 描画
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    plt.tick_params(
        axis="both",
        which="both",
        bottom="off",
        top="off",
        labelbottom="off",
        right="off",
        left="off",
        labelleft="off",
    )

    (line,) = ax.plot([0.5], [2.5], marker="o", color="g", markersize=60)
    return fig, line


def main(model_no, n_iter, is_plot, verbose):

    # fig, line = create_maze()

    # 初期の方策を決定するパラメタ\theta_0を設定
    # 行は状態0~7, 列は移動方向で 🖕👉👇👈
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
        model = PolicyGradientMethodModel(theta_0, n_iter=n_iter, verbose=verbose)

    print(f"model no {model_no}")

    # 実行
    s_a_history = model.run()

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
    args = sys.argv

    is_plot = False
    verbose = False
    model = int(args[1])
    n_iter = int(args[2])
    main(model, n_iter, is_plot, verbose)
