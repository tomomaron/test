from HRT.field import O, W, G, M, STATE_MAP
import numpy as np
from enum import Enum
import random


class QLearningAgent:
    def __init__(self, field, start_pos, learning_parameters):
        self.x, self.y = start_pos
        self.x_start, self.y_start = start_pos

        if field[self.y][self.x] != O:
            raise ValueError("The start position must be in a way.")

        self.field = np.asarray(field)
        self.field_height, self.field_width = self.field.shape
        self.parameters = learning_parameters
        self.q_table = np.full((self.get_state_counts(), 4), self.parameters.q_init_value)  # init Q table

    def run(self):
        """
        学習を開始する
        :return: 最良step長
        """
        print("Start learning the agent. ", end="")
        print("(Max trials: {}, Max steps: {})".format(self.parameters.max_trial, self.parameters.max_step))
        min_step_count = self.parameters.max_step

        for trial in range(self.parameters.max_trial):
            self.init_position()

            for step in range(self.parameters.max_step):
                if self.run_step():
                    if step < min_step_count:
                        min_step_count = step
                        print("Updated the best (on {} trials): {}".format(trial, min_step_count))

                    if self.parameters.show_step_details:
                        print("Trials: {}, Steps: {}".format(trial, step))

                    break
        print()

        return min_step_count

    def run_step(self):
        """
        学習を1ステップ進める
        :return: ステップの結果ゴールしたか否か
        """
        old_x, old_y = self.x, self.y
        old_state = self.get_state()

        action = self.choice_action()
        self.act(action)

        reward = self.get_reward()

        if self.parameters.show_step_details:
            print("------------------------------------------")
            self.print_field()
            self.print_state()

        return self.is_goal

    def print_field(self):
        """
        現在のマップ状況を描画する
        """
        filed_copy = self.field.copy()
        filed_copy[self.y][self.x] = M
        for row in filed_copy:
            for cell in row:
                print(STATE_MAP[cell], end="")
            print()

    def print_state(self):
        """
        現在のstateを表示する
        """
        expression = self.parameters.state_expression
        current_state = self.get_state()

        if expression == Parameters.StateExpression.COORDINATE:
            print("Current state = Position({}, {}), Raw value({})".format(self.x, self.y, current_state))
            print("UP    : {}".format(self.q_table[current_state][0]))
            print("DOWN  : {}".format(self.q_table[current_state][1]))
            print("LEFT  : {}".format(self.q_table[current_state][2]))
            print("RIGHT : {}".format(self.q_table[current_state][3]))

        elif expression == Parameters.StateExpression.NEIGHBORHOOD:
            print("Current State = Raw({})".format(current_state))
            print("UP    : {}".format(self.q_table[current_state][0]))
            print("DOWN  : {}".format(self.q_table[current_state][1]))
            print("LEFT  : {}".format(self.q_table[current_state][2]))
            print("RIGHT : {}".format(self.q_table[current_state][3]))

            top_left = (current_state >> 14) & 0b11
            top = (current_state >> 12) & 0b11
            top_right = (current_state >> 10) & 0b11
            left = (current_state >> 8) & 0b11
            right = (current_state >> 6) & 0b11
            bottom_left = (current_state >> 4) & 0b11
            bottom = (current_state >> 2) & 0b11
            bottom_right = current_state & 0b11

            print("{}{}{}".format(STATE_MAP[top_left], STATE_MAP[top], STATE_MAP[top_right]))
            print("{}○{}".format(STATE_MAP[left], STATE_MAP[right]))
            print("{}{}{}".format(STATE_MAP[bottom_left], STATE_MAP[bottom], STATE_MAP[bottom_right]))

        else:
            raise NotImplementedError

    def act(self, action):
        """
        渡されたActionオブジェクトにしたがってエージェントが行動する．
        移動先が壁やマップ外の場合は何も変化しない．
        :param action: 次の行動を示すActionオブジェクト
        :return: 移動の成功可否
        """
        x_after_act = self.x + action.value[0]
        y_after_act = self.y + action.value[1]

        try:
            if self.field[y_after_act][x_after_act] == W:
                raise IndexError

            self.x, self.y = x_after_act, y_after_act
            return True

        except IndexError:
            return False

    def init_position(self):
        """
        エージェントを初期位置に戻す
        :return:
        """
        self.x, self.y = self.x_start, self.y_start

    def choice_action(self):
        """
        Qテーブルを元に各戦略に従って次のactionを決定する
        :return: action(Action)
        """
        # TODO: 各戦略におけるActionの選択方法を実装する
        # 各戦略で選択後，Directionクラスのenumを返却する
        strategy = self.parameters.action_choice_strategy

        if strategy == Parameters.ActionChoiceStrategy.GREEDY:
            # TODO: --------------- greedy法 ------------------

            return random.choice(list(Action))  # ←はgreedy法には関係ありません

            # TODO: -------------------------------------------

        elif strategy == Parameters.ActionChoiceStrategy.E_GREEDY:
            # TODO: -------------- e-greedy法 -----------------

            pass

            # TODO: -------------------------------------------

        elif strategy == Parameters.ActionChoiceStrategy.ROULETTE:
            # TODO: -------------- ルーレット選択 ----------------

            pass

            # TODO: -------------------------------------------

        else:
            raise NotImplementedError

    def get_reward(self):
        """
        報酬を返却する
        :return: 報酬
        """
        # TODO: --------------- 報酬を計算する ---------------
        # ゴールしていれば self.parameters.reward を返却する．そうでなければ0．

        return 0.0

        # TODO: -------------------------------------------

    def update_q_table(self):
        """
        Qテーブルを更新する
        """
        # TODO: ------------ Qテーブルを更新する -------------

        pass

        # TODO: -------------------------------------------

    def get_state(self):
        """
        現在のstateを返却する
        :return: state
        """
        # TODO:
        expression = self.parameters.state_expression

        if expression == Parameters.StateExpression.COORDINATE:
            return self.y * self.field_width + self.x

        elif expression == Parameters.StateExpression.NEIGHBORHOOD:
            # TODO: ---------- 近傍８マスを状態とする時 ------------
            # 異なる状態が一意に表現できるようにする．ビット演算を利用する．
            # print_state()のフォーマットを参考にする

            pass

            # TODO: -------------------------------------------

        else:
            raise NotImplementedError

    def get_state_counts(self):
        """
        stateの取りうるパターン数を返却する
        :return: stateの取りうるパターン数
        """
        if self.parameters.state_expression == Parameters.StateExpression.COORDINATE:
            return self.field_height * self.field_width  # マップのマス数

        elif self.parameters.state_expression == Parameters.StateExpression.NEIGHBORHOOD:
            return 4 ** 8  # 4（State）^ 8（近傍マス）

        else:
            raise NotImplementedError

    @property
    def is_goal(self):
        return self.field[self.y][self.x] == G


class Action(Enum):
    """
    行動を示すEnumオブジェクト
    Action.UP.valueのように.valueで値にアクセスできる
    """
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


class Parameters:
    """
    学習パラメータ用クラス
    """
    class ActionChoiceStrategy(Enum):
        GREEDY = 0  # Greedy法
        E_GREEDY = 1  # ε-Greedy法
        ROULETTE = 2  # ルーレット選択

    class StateExpression(Enum):
        COORDINATE = 0  # 絶対座標
        NEIGHBORHOOD = 1  # 近傍八マス

    def __init__(
            self,
            action_choice_strategy,  # 行動選択の戦略
            state_expression,  # stateの表現方法
            max_trial,  # 学習における最大trial数
            max_step,  # 1trialあたりの最大step数
            show_step_details,  # 学習経過を表示するか否か
            reward,  # ゴール時の報酬量
            q_init_value  # Qテーブルの初期値
    ):
        self.action_choice_strategy = action_choice_strategy
        self.state_expression = state_expression
        self.max_trial = max_trial
        self.max_step = max_step
        self.show_step_details = show_step_details
        self.reward = reward
        self.q_init_value = q_init_value
