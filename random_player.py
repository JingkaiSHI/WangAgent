import random
from my_player3 import parse_input, write_move, get_all_legal_moves

class RandomPlayer:
    def __init__(self):
        self.player = None
        self.prev_board = None
        self.curr_board = None

    def load_cur_state(self, input_file="input.txt"):
        self.player, self.prev_board, self.curr_board = parse_input(input_file)

    def select_move(self):
        legal_moves = get_all_legal_moves(self.curr_board, self.prev_board, self.player)

        if not legal_moves:
            return "PASS"
        print(legal_moves)
        filtered_moves = [move for move in legal_moves if move != "PASS" and not self._is_fully_surrounded(move)]
        if not filtered_moves:
            return "PASS"
        return random.choice(filtered_moves)

    def _is_fully_surrounded(self, move):
        if move == "PASS":
            return False
        i, j = move
        directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        for dx, dy in directions:
            x, y = i + dx, j + dy
            if 0 <= x < 5 and 0 <= y < 5:
                if self.curr_board[x][y] == 0:
                    return False
        return True
    
    def play(self):
        try:
            self.load_cur_state("test_input_1.txt")
            chosen_move = self.select_move()
            write_move(chosen_move)
        except Exception as e:
            write_move("PASS")
            raise


if __name__ == "__main__":
    player = RandomPlayer()
    player.curr_board = [
        [0,0,0,0,0],
        [0,1,2,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0]]
    player.prev_board = player.curr_board
    player.player = 1
    print("random select a point:", player.select_move())

    surrounded_board = [
        [0,2,0,0,0],
        [2,0,2,0,0],
        [2,2,2,2,2],
        [2,2,2,2,2],
        [2,2,2,2,2]]
    player.curr_board = surrounded_board
    print("forced selection under all surrounding while testing case of suicide rule:", player.select_move())

    RandomPlayer().play()
    with open("output.txt") as f:
        print("move selected:", f.read())