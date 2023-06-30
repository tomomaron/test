import os
from itertools import chain
from math import sqrt
import numpy as np

from NN import NN

MAX_ITER = 30000  # 学習回数
MIN_ERROR = 0.001  # 教師出力データとＮＮの出力データの誤差の合計値の基準値
HIDDEN = 20  # 中間ノード数


def read_data(file_name):
    """
    データの読み込み
    :param file_name: 読み込むファイル名
    :returns (学習枚数, 入力数, 出力数, 入力データリスト, 出力データリスト)

    教師データ(teacher20.txt)のフォーマット
        学習枚数 N
        入力数 in
        出力数 out
        (1枚目の情報)
        (2枚目の情報)
        …
        (N枚目の情報)
    """
    with open(file_name, "r") as f:
        lines = f.readlines()
        lines = [l.rstrip() for l in lines]  # 改行除去

    data_num, in_num, out_num = map(int, lines[:3])  # 学習枚数，入力数，出力数の読み取り
    data_size = int(sqrt(in_num))
    lines = lines[3:]

    print("学習枚数 :", data_num)
    print("入力数 :", in_num)
    print("出力数 :", out_num)

    # 学習データの読み取り
    in_arrays = []
    out_arrays = []
    for i in range(data_num):
        # ファイル名の読み込み
        data_name = lines[0]
        print(data_name)

        # 入力データの読み込み
        data = chain.from_iterable([line.split(" ") for line in lines[1:1 + data_size]])
        data = list(map(int, data))
        in_array = np.array(data)
        # 出力データの読み込み(0 or 1)
        data = lines[1 + data_size + 2].split(" ")
        data = list(map(int, data))
        out_array = np.array(data)

        # 過学習を防ぐため，入力値を調整する
        np.where(in_array == 0, 0.1, 0.9)

        in_arrays.append(in_array)
        out_arrays.append(out_array)
        lines = lines[1 + data_size + 4:]

    return data_num, in_num, out_num, in_arrays, out_arrays


def learn():
    """NNの学習部分"""
    # 教師データを読み込む
    # デフォルトはteacher20.txt
    while True:
        print("教師データファイル(.txt): ", end="")
        file_name = input()
        if os.path.exists(file_name):
            _, in_num, out_num, in_arrays, out_arrays = read_data(file_name)
            break
        else:
            print("ファイルが見つかりません")

    # NNオブジェクトの生成
    nn = NN(in_num, HIDDEN, out_num)
    # 状態・結合荷重の初期化
    nn.init_state()  # ノードの状態の初期化【作成部分】
    nn.init_weight()  # 結合荷重の初期化【作成部分】

    # ここからBP法による学習を行う
    # 一定回数以下あるいは，出力との一定誤差以上の場合は繰り返す
    for i in range(MAX_ITER):
        # NNを誤差逆伝播法で学習させる【作成部分】
        nn.back_propagation(in_arrays, out_arrays)
        # 誤差の合計を算出
        error = nn.calc_error(in_arrays, out_arrays)

        if i % 100 == 0:
            print("<iter {}> : Error = {}".format(i, error))

        if error < MIN_ERROR or i == MAX_ITER:
            print("学習終了 : <iter {}> : Error = {}".format(i, error))
            break

    while True:
        print("NN出力ファイル名(*.npz): ", end="")
        file_name = input()
        if os.path.splitext(file_name)[1] == ".npz":
            break
        else:
            print("出力ファイル名が適切ではありません")

    # 学習結果をファイルに保存する
    nn.save_nn(file_name)
    print("{} : NN学習データ書き込み終了".format(file_name))


def experiment():
    """未知への適用"""
    # 未知データファイルを読み込む
    # デフォルトはcheck20.txt
    while True:
        print("未知データファイル(.txt): ", end="")
        data_file_name = input()
        if os.path.exists(data_file_name):
            data_num, in_num, out_num, in_arrays, out_arrays = read_data(data_file_name)
            break
        else:
            print("ファイルが見つかりません")

    while True:
        print("NN学習ファイルを指定してください(*.npz): ", end="")
        nn_file_name = input()
        if os.path.exists(nn_file_name):
            break
        else:
            print("ファイルが見つかりません")

    # NNオブジェクトの生成
    nn = NN(in_num, HIDDEN, out_num)
    # ファイルからNNを読み込む
    nn.load_nn(nn_file_name)

    right_num = 0
    for i in range(data_num):
        # NNに入力データを入力して出力(認識率含む）を得る(ファイル出力）【作成部分】
        result = nn.calc_nn(in_arrays[i])
        # 正解しているかどうかの判定
        if out_arrays[i][result] == 1:
            right_num += 1

    print("正解率{}({}枚/{}枚)".format(right_num * 100 / data_num, right_num, data_num))
    print("未知への適用終了")


def main():
    print("///////////////////////Program Start///////////////////////")

    while True:
        print("program:")
        print("1.学習")
        print("2.適用")
        print("3.終了")
        print("選択してください(1 or 2 or 3): ", end="")

        mode = int(input())
        if mode == 1:
            print("学習")
            learn()
            print("適用")
            experiment()
        elif mode == 2:
            print("適用")
            experiment()
        elif mode == 3:
            break
        print()

    print("////////////////終了////////////////")


if __name__ == "__main__":
    main()
