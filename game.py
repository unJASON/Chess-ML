# -*- coding: utf-8 -*-
"""
@author: Junxiao Song
"""

from __future__ import print_function
import numpy as np


class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """

        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            # print(moves)
            # print(players)
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            # print(move_curr)
            # print(move_oppo)
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        # print(square_state[0, ::-1, :])   #当前的人所下的点
        # print(square_state[1, ::-1, :])   #对手所下的点
        # print(square_state[2, ::-1, :])   #最后一步落子
        # print(square_state[3, ::-1, :])   #先后手
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move
    def piecesCount(self,border,states,x,y,player, pieces_count, x1, y1):
        for i in range(1, self.n_in_row):
            new_x = x + x1*i
            new_y = y + y1*i
            #在边界内部
            if new_x < border and new_y < border and new_x >=0 and new_y >=0:
                if states.__contains__(new_x*border+new_y) and states[new_x*border+new_y] == player:
                    pieces_count +=1
                else:
                    break
            else:
                break
        return pieces_count

    def coorJudge(self,border,x,y,player, states):

        pieces_count = 0
        pieces_count = self.piecesCount(border,states,x,y,player, pieces_count, 1, 0)  # 右边
        pieces_count = self.piecesCount(border,states,x,y,player, pieces_count, -1, 0)  # 左边
        if pieces_count >= self.n_in_row - 1:
            return True

        pieces_count = 0
        pieces_count = self.piecesCount(border,states,x,y,player, pieces_count, 0, -1)  # 上边
        pieces_count = self.piecesCount(border,states,x,y,player, pieces_count, 0, 1)  # 下边
        if pieces_count >= self.n_in_row - 1:
            return True

        pieces_count = 0
        pieces_count = self.piecesCount(border,states,x,y,player, pieces_count, 1, 1)  # 右下角
        pieces_count = self.piecesCount(border,states,x,y,player, pieces_count, -1, -1)  # 左上角
        if pieces_count >= self.n_in_row - 1:
            return True

        pieces_count = 0
        pieces_count = self.piecesCount(border,states,x,y,player, pieces_count, 1, -1)  # 右上角
        pieces_count = self.piecesCount(border,states,x,y,player, pieces_count, -1, 1)  # 左下角
        if pieces_count >= self.n_in_row - 1:
            return True

        return False



    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row + 2:
            return False, -1
        width = self.width
        states = self.states
        m = self.last_move
        x = m // width
        y = m % width
        player = states[m]
        if self.coorJudge(width,x,y,player,states):
            return True,player
        return False, -1

        # for m in moved:
        #     h = m // width
        #     w = m % width
        #     player = states[m]
        #
        #     if (w in range(width - n + 1) and
        #             len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
        #         return True, player
        #
        #     if (h in range(height - n + 1) and
        #             len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
        #         return True, player
        #
        #     if (w in range(width - n + 1) and h in range(height - n + 1) and
        #             len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
        #         return True, player
        #
        #     if (w in range(n - 1, width) and h in range(height - n + 1) and
        #             len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
        #         return True, player
        # return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        if self.last_move == -1:
            return False, -1
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        #for i in range(height - 1, -1, -1):
        for i in range(height):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
        # if True:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
            # if True:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
if __name__ == '__main__':
    board_width = 15
    board_height = 15
    n_in_row = 5
    board = Board(width=board_width,
                       height=board_height,
                       n_in_row=n_in_row)
    game = Game(board)

    board.init_board()
    board.do_move(0)
    board.do_move(5)
    board.do_move(1)
    board.do_move(6)
    board.do_move(2)
    board.do_move(7)
    board.do_move(3)
    board.do_move(8)
    board.do_move(4)
    print(board.has_a_winner())
    print(game.graphic(game.board,1,2))
