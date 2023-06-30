import numpy as np


class NN:
    ALPHA = 0.1
    BETA = 0.1

    def __init__(self, in_num, hid_num, out_num):
        self.input_state = np.empty(in_num + 1)  # +1はバイアス
        self.hidden_state = np.empty(hid_num + 1)  # +1はバイアス
        self.output_state = np.empty(out_num)
        self.weight_ih = np.empty([in_num + 1, hid_num])  # input, hidden間の重み
        self.weight_ho = np.empty([hid_num + 1, out_num])  # hidden, output間の重み

    def init_state(self):
        """
        NNのノード状態の初期化
        入力ノードと中間ノードはバイアスに注意
        """
        # 【作成部分】
        # 入力ノードの初期化

        # バイアス入力(状態遷移部分に入れても良い)

        # 中間ノードの初期化

        # バイアス入力(状態遷移部分に入れても良い)

        # 出力ノードの初期化
        pass

    def init_weight(self):
        """
        NNの結合荷重の初期化(※乱数による初期化！)
        """
        # 【作成部分】
        # 入力ー中間ノード間結合荷重の初期化

        # 中間ー出力ノード間結合荷重の初期化
        pass

    def back_propagation(self, teacher_in, teacher_out):
        """
        誤差逆伝播法
        :param teacher_in: 教師の入力データ、一次元numpy配列(,400)のリスト
        :param teacher_out: 教師の出力データ、一次元numpy配列（,2)のリスト
        """
        # 【作成部分】(理論的な説明はテキストなどを参考にしてください)
        # 作成例 == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
        # for文部分を一部記入した例です。
        # コメントアウトをはずしてそのまま使用してもかまいません
        # for文が入っているとかえってやり辛い / 自分で組みたい / ネットのソースをとってきたい人は無視してかまいません。
        # TODO: サンプルを作成する
        pass

    def calc_error(self, teacher_in, teacher_out):
        """
        ノードの状態を遷移させていき教師データとNNの出力データの二乗誤差の合計を算出する
        :param teacher_in: 教師の入力データ、一次元numpy配列(,400)のリスト
        :param teacher_out: 教師の出力データ、一次元numpy配列（,2)のリスト
        :return: 教師出力データとNNの出力データの二乗誤差の合計
        """
        error = 0.0
        for i, input_data in enumerate(teacher_in):
            self.calc_state(input_data)  # 状態遷移を行う
            error += np.power((teacher_out[i] - self.output_state), 2).sum()  # 出力から二乗誤差を算出

        return error / 2

    def calc_state(self, input_data):
        """
        ノードの状態を遷移させる
        :param input_data: 入力データ
        """
        # 【作成部分】
        # ノード状態の初期化

        # 入力ノードに入力を入れる

        # 状態遷移
        # 中間ノードの状態を算出
        # ①入力を計算(Σ(入力ユニットの値 * 結合荷重))

        # ②出力関数(シグモイド関数)で出力を計算

        # 出力ノードの状態を算出
        # ①入力を計算(Σ(中間ユニットの値 * 結合荷重))

        # ②出力関数(シグモイド関数)で出力を計算
        pass

    def save_nn(self, file_name):
        """
        NNをファイルに保存する
        :param file_name: 保存するファイル名(拡張子は.npzでなければならない)
        """
        np.savez(
            file_name,
            weight_ih=self.weight_ih,
            weight_ho=self.weight_ho)

    def load_nn(self, file_name):
        """
        NNをファイルから読み込む
        :param file_name: 保存するファイル名(拡張子は.npzでなければならない)
        """
        npz = np.load(file_name)
        self.weight_ih = npz["weight_ih"]
        self.weight_ho = npz["weight_ho"]

    def print_nn(self):
        """NNの結合荷重を表示"""
        # TODO: もう少しきれいに整形して表示したほうが良いかも
        print(self.weight_ih)
        print()
        print(self.weight_ho)
        print()

    def calc_nn(self, input_data):
        """
        入力データに対する出力データを算出して，認識結果を指定したファイルに出力する
        認識率も出力する
        :return 認識結果
        """
        self.calc_state(input_data)
        for i in range(self.output_state.size):
            result = "[{}]=>{}".format(i, self.output_state[i])
            print(result)

        max_index = np.argmax(self.output_state)
        print("適用結果:{}({})".format(max_index, self.output_state[max_index]))
        print("********************")

        return max_index
