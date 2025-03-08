def parse_input(input_file="input.txt"):
    """
    Parse the input file for board and state for current piece type and board state
    :param input_file: a file directory
    :return: (piece_type, prev_board_after_player_action, board_after_opponent_action)
    """
    try:
        with open(input_file, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
            assert len(lines) == 11
            piece_type = int(lines[0])
            assert piece_type == 1 or piece_type == 2
            prev_board = []
            for line in lines[1:6]:
                assert len(line) == 5
                prev_board.append([int(c) for c in line])
            cur_board = []
            for line in lines[6:11]:
                assert len(line) == 5
                cur_board.append([int(c) for c in line])
            return piece_type, prev_board, cur_board
    except FileNotFoundError:
        raise FileNotFoundError("input not found")
    except Exception as e:
        raise RuntimeError(f"Input Parsing Failed: {str(e)}")

def write_move(move):
    try:
        with open("output.txt", 'w') as f:
            if move == "PASS":
                f.write("PASS")
            elif isinstance(move, tuple) and len(move) == 2:
                i, j = move
                if 0 <= i < 5 and 0 <= j < 5:
                    f.write(f"{i},{j}")
                else:
                    raise ValueError("Invalid move coordinates!")
    except Exception as e:
        raise RuntimeError(f"Failed to write output: {str(e)}")


def count_liberties(board, i, j, player):
    """
    count the liberty order of piece (or group) at position (i, j)
    :param board: the current board of size 5 x 5
    :param i: ith row
    :param j: jth column
    :param player: piece_type
    :return: liberty order of piece or group at position (i, j)
    """
    if board[i][j] != player:
        # it is either empty or opponent's piece
        return 0

    visited = set()
    stack = [(i, j)]
    liberties = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while stack:
        x, y = stack.pop()
        if (x, y) in visited:
            continue
        visited.add((x, y))
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 5 and 0 <= y < 5:
                if board[nx][ny] == 0:
                    liberties += 1
                elif board[nx][ny] == player:
                    stack.append((nx, ny))
    return liberties

def is_suicide(board, i, j, player):

################################### Definition of Agent ##################################
# doesn't need to worry about other util functions, only worry about how things are calculated
if __name__ == "__main__":
    test_input_1 = [
        "1",
        "00000", "00000", "00000", "00000", "00000",  # prev_board
        "00000", "00000", "00000", "00000", "00000"  # curr_board
    ]

    # 测试案例2：已有棋子的棋盘
    test_input_2 = [
        "2",
        "00110", "00210", "00200", "02000", "00000",  # prev_board
        "00110", "00210", "00200", "02010", "00000"  # curr_board
    ]

    # 生成测试文件
    with open("test_input_1.txt", 'w') as f:
        f.write("\n".join(test_input_1))

    # 解析测试
    try:
        player, prev_board, curr_board = parse_input("test_input_1.txt")
        print(f"Player: {player}")
        print("Previous Board:")
        for row in prev_board: print(row)
        print("\nCurrent Board:")
        for row in curr_board: print(row)

        # 测试写入
        write_move((2, 3))  # 正常坐标
        write_move("PASS")  # 放弃
        # write_move((5,0))  # 会触发异常
    except Exception as e:
        print(f"Test Error: {e}")