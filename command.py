from game import Board,Game
from human_play import Human
from mcts_alphaZero import MCTSPlayer
from policy_value_net_15 import PolicyValueNet

if __name__ == '__main__':
    n = 5
    board_width, board_height = 15, 15
    n_playout = 800  # num of simulations for each move
    c_puct = 5  #UCT formula parameters
    init_model = './best_policy.model'
    board = Board(width=board_width, height=board_height, n_in_row=n)
    game = Game(board)
    human1 = Human()

    policy_value_net = PolicyValueNet(board_width,
                                      board_height,
                                      model_file=init_model)
    mcts = MCTSPlayer(policy_value_net.policy_value_fn,
                      c_puct=c_puct,
                      n_playout=n_playout,
                      is_selfplay=0)
    game.start_play(human1,mcts,start_player=1,is_shown=1)