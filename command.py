from game import Board,Game
from human_play import Human

from policy_value_net_tensorflow import PolicyValueNet
n = 5
width, height = 8, 8
if __name__ == '__main__':
    board = Board(width=width, height=height, n_in_row=n)
    game = Game(board)
    #human1 = Human()
    #human2 = Human()
    #game.start_play(human1,human2,start_player=1,is_shown=1)