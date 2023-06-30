W = 0  # Wall
O = 1  # Way
G = 2  # Goal
M = 3  # Me

STATE_MAP = {
    W: "■",
    O: "　",
    G: "Ｇ",
    M: "○"
}

field = [
    [W, W, W, W, W, W, W, W, W, W],
    [W, O, W, W, W, W, W, W, W, W],
    [W, O, W, W, W, W, W, W, W, W],
    [W, O, W, W, W, W, W, W, W, W],
    [W, O, O, O, O, O, W, W, W, W],
    [W, W, W, W, W, O, W, W, W, W],
    [W, W, W, W, W, O, W, W, W, W],
    [W, W, W, W, W, O, W, W, W, W],
    [W, W, W, W, W, O, O, O, G, W],
    [W, W, W, W, W, W, W, W, W, W]
]

field2 = [
    [W, W, W, W, W, W, W, W, W, W],
    [W, O, O, O, W, O, O, W, O, W],
    [W, O, W, O, O, O, W, O, O, W],
    [W, O, W, W, O, W, O, W, O, W],
    [W, O, O, O, O, O, O, W, O, W],
    [W, W, W, W, W, W, O, O, O, W],
    [W, O, W, W, W, W, W, O, W, W],
    [W, O, O, O, W, W, O, O, W, W],
    [W, G, W, O, O, O, O, W, W, W],
    [W, W, W, W, W, W, W, W, W, W],
]
