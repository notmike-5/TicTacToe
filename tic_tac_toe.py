'''
Tic-Tac-Toe w/ some simple agents.

Public Classes:
    - TicTacToe: create a new instance of a Tic-Tac-Toe game
    Available agents:
        - RandomPlayer:
            This player chooses any available move and takes it.
        - AlphaBetaPlayer:
            This player uses the minimax algorithm with α-β pruning.
'''

import random
import numpy as np


class Player():
    '''
    Hate the game, don't hate the player.
    '''

    def __init__(self, game, mark: str, name: str = 'HumanPlayer'):
        self.game = game
        self.mark = mark
        self.name = name
        self.record = np.array([0, 0, 0])  # Win-Lose-or-Draw

    def get_move(self) -> tuple[int, int]:
        '''
        What you tryin' to do!?
        '''
        pass

    def __repr__(self) -> str:
        '''
        Represent the player's record.
        '''
        return f'{self.name}\t {self.mark}s\t{self.record}'


def scoreboard(player: Player, agent: Player) -> None:
    '''
    Show the current round score.
    '''
    print("\n##########\nSCOREBOARD\n##########\n")
    print("Player\t\tMark\t[W-L-D]")
    print("------\t\t----\t-------")
    print(player)
    print(agent)

    return None


def play_game(game, player='human', agent: Player = None, n: int = 100) -> bool:
    '''
    Play / Simulate games between various agents.
    '''
    if agent is None:
        print("No agent provided!")
        return False
    game = game(player, agent)

    if game.player.name == 'HumanPlayer':
        good_resp = ['y', 'ye', 'yes', 's', 'si', 'sim',
                     't', 'true', '1', 'on', 'c', 'continue']
        resp = 'yes'
        while resp in good_resp:
            game.player.record += game.play()

            scoreboard(game.player, game.agent)

            try:
                resp = input("\nWould you like to play again? ").lower()
            except ValueError:
                print('Huh?', end=' ')
                continue
        # fin
        print('\nk, bye!')
    else:
        # simulate n games
        for _ in range(n):
            game.play()

        scoreboard(game.player, game.agent)

    return True


class TicTacToe():
    '''
    tic-tac-toe: a simple model for playing tic-tac-toe with agents.
    '''
    AI_PLAYER, HUMAN_PLAYER = 'X', 'O'
    WIN, LOSS, DRAW = 1000, -1000, 0
    INIT_DEPTH = 0
    EMPTY = ' '

    def __init__(self, player: Player = 'human', agent: Player = None, size: int = 3) -> object:
        self.board = TicTacToe.init_board(self)
        self.size = size

        # human init
        if player == 'human':
            try:
                # give human player choice of mark
                if input('Xs or Os? ')[0].upper() == 'X':
                    self.HUMAN_PLAYER, self.AI_PLAYER = 'X', 'O'
            except ValueError:
                pass

            self.player = Player(game=self, mark=self.HUMAN_PLAYER)

        else:  # (opt.) agent_2 init
            try:
                self.player = player(game=self, mark=self.HUMAN_PLAYER)
            except ValueError:
                print(f'{type(player)} agent init failed.')
                return

        # agent init
        if agent is None:
            print("No agent provided!")
            return
        try:
            self.agent = agent(game=self, mark=self.AI_PLAYER)
        except ValueError:
            print(f'{type(agent)} agent init failed.')
            return

        spaces = [(i, j) for i in range(self.size) for j in range(self.size)]
        self.idxs = dict(enumerate(spaces, 1))
        self.spaces = {v: k for k, v in self.idxs.items()}
        self.WIN_STATES = {
            # TODO: can get funky here. e.g. connect-k on an n x m board
            # cols
            frozenset((i, 0) for i in range(self.size)),
            frozenset((i, 1) for i in range(self.size)),
            frozenset((i, 2) for i in range(self.size)),
            # rows
            frozenset((0, j) for j in range(self.size)),
            frozenset((1, j) for j in range(self.size)),
            frozenset((2, j) for j in range(self.size)),
            # diagonals
            frozenset((i, i) for i in range(self.size)),
            frozenset((i, size-i-1) for i in range(self.size))
        }

    def get_space(self, index: int) -> tuple[int, int]:
        '''
        Convert index to space.
        '''
        return self.idxs[index]

    def get_index(self, row: int, col: int) -> int:
        '''
        Convert space to index.
        '''
        return self.spaces[(row, col)]

    def getc(self, row: int, col: int) -> str:
        '''
        Helper to print pretty boards.
        '''
        c = self.board[row, col]
        return c if c != self.EMPTY else self.get_index(row, col)

    # TODO: generalize to an n x m board
    def print_board(self, fancy: bool = False) -> None:
        '''
        Print the current board state.
            param: fancy, will show labels over available spaces
        '''
        # overlay space labels [1-9]
        if fancy:
            print(f'{self.getc(0,0)} | {self.getc(0,1)} | {self.getc(0,2)}')
            print("----------")
            print(f'{self.getc(1,0)} | {self.getc(1,1)} | {self.getc(1,2)}')
            print("----------")
            print(f'{self.getc(2,0)} | {self.getc(2,1)} | {self.getc(2,2)}\n')

            return None

        # just print current board
        print(f'{self.board[0,0]} | {self.board[0,1]} | {self.board[0,2]}')
        print("----------")
        print(f'{self.board[1,0]} | {self.board[1,1]} | {self.board[1,2]}')
        print("----------")
        print(f'{self.board[2,0]} | {self.board[2,1]} | {self.board[2,2]}\n')

        return None

    def board_full(self) -> bool:
        '''
        Check the board and report if it is full.
        '''
        return not (self.board == self.EMPTY).any()

    def match_outcome(self, state: int, verbose: bool = True) -> np.array:
        '''
        Helper to print the outcome of matches
        '''
        if not verbose:
            match state:
                case self.WIN:
                    return np.array([1, 0, 0])
                case self.LOSS:
                    return np.array([0, 1, 0])
                case _:  # DRAW
                    return np.array([0, 0, 1])

        # for humans (pitiful)
        match state:
            case self.WIN:
                print("\nCongrats, you won! :D")
                return np.array([1, 0, 0])
            case self.LOSS:
                print("\nSorry, you lost. :'(")
                return np.array([0, 1, 0])
            case _:  # DRAW
                print("\nThe game was a draw. -_-")
                return np.array([0, 0, 1])

    def init_board(self, n: int = 3) -> np.array:
        '''
        Initialize a new game board.
        '''

        return np.full((n, n), self.EMPTY)

    def get_legal_moves(self) -> set:
        '''
        Get the available moves.
        '''
        idx = np.where(self.board == self.EMPTY)

        return {(idx[0][i], idx[1][i]) for i in range(len(idx[0]))}

    def get_occupied_spaces(self, player: str) -> set:
        '''
        Get the set of occupied spaces.
        '''
        idx = np.where(self.board == player)

        return {(idx[0][i], idx[1][i]) for i in range(len(idx[0]))}

    def get_opponent(self, player: str) -> str:
        '''
        Get opponent's marker
        '''
        if player == 'X':
            return 'O'
        if player == 'O':
            return 'X'

        raise ValueError("Problem in get_opponent()")

    def is_occupied(self, space: set) -> bool:
        '''
        Check if space is occupied.
        '''
        return self.get_legal_moves().isdisjoint(space)

    def board_state(self, player: str) -> int:
        '''
        Get the state of the board.
        '''
        # check if player has won
        occ = self.get_occupied_spaces(player)
        if self.game_won(occ):
            return self.WIN

        # check if opponent has won
        opp = self.get_opponent(player)
        occ = self.get_occupied_spaces(opp)
        if self.game_won(occ):
            return self.LOSS

        return self.DRAW

    def game_won(self, occupied: set) -> bool:
        '''
        Determine if the game is won given the occupied spaces.
        occupied: set of *a player's* occupied spaces
        '''
        for s in self.WIN_STATES:
            if len(s.intersection(occupied)) >= 3:
                return True

        return False

    def game_finished(self) -> bool:
        '''
        Check board, and report if the game is complete.
        '''
        if self.board_full() or self.board_state(self.AI_PLAYER) != self.DRAW:
            return True

        return False

    def play(self, n: int = 100):
        '''
        Play some rounds of TicTacToe.
        '''
        if self.player.name == 'HumanPlayer':
            print("#######################")
            print("Let's Play tic-tac-toe!")
            print("#######################")
            print(
                "For ease-of-use the board\nhas been keyed to 10-key [1-9]\n")
            self.board = self.init_board()
            self.print_board(fancy=True)

            # Let's Play tic-tac-toe! (Finally!)
            while not self.game_finished():
                try:
                    space = int(input("Which space? "))
                except ValueError:
                    print('Huh?', end=' ')
                    continue
                if space not in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
                    print('Huh?', end=' ')
                    continue

                (row, col) = self.get_space(space)

                if self.is_occupied({(row, col)}):
                    print("Position is occupied! Pick again.\n")
                    continue

                self.board[row][col] = self.HUMAN_PLAYER

                if not self.game_finished():
                    # TODO: use best_score or lose it
                    (row, col) = self.agent.get_move()
                    self.board[row][col] = self.AI_PLAYER
                    self.print_board()

            self.print_board()

            # keep score
            self.agent.record += self.match_outcome(
                self.board_state(self.AI_PLAYER), False)
            return self.match_outcome(self.board_state(self.HUMAN_PLAYER))

        # robot -vs- robot!
        while not self.game_finished():
            (row, col) = self.player.get_move()
            self.board[row][col] = self.HUMAN_PLAYER

            if not self.game_finished():
                (row, col) = self.agent.get_move()
                self.board[row][col] = self.AI_PLAYER

        # keep score
        self.player.record += self.match_outcome(
            self.board_state(self.HUMAN_PLAYER), False)
        self.agent.record += self.match_outcome(
            self.board_state(self.AI_PLAYER), False)


class RandomPlayer(Player):
    '''
    This is just a random player.
    '''

    def __init__(self, game: TicTacToe, mark: str) -> Player:
        super().__init__(game, mark, name='RandomPlayer')

    def get_move(self) -> tuple[int, int]:
        '''
        The agent takes any available move.
        '''
        move = random.sample(list(self.game.get_legal_moves()), 1)[0]
        return move


class AlphaBetaPlayer(Player):
    '''
    This is a Min-Max agent with Alpha-Beta pruning.
    '''

    def __init__(self, game: TicTacToe, mark: str):
        super().__init__(game, mark, name='AlphaBetaPlayer')

    def get_move(self) -> tuple[int, int]:
        '''
        Return the agent's move according to decision function.
        '''
        _, best_move = self.minimax(
            self.game.AI_PLAYER, self.game.INIT_DEPTH, self.game.LOSS, self.game.WIN)

        return best_move

    def minimax(self, player: str, depth: int, alpha: float, beta: float) -> tuple[int, tuple[int, int]]:
        '''
        Implement the minimax algorithm w/ αlphα-βeta pruning to play optimal strategy (we hope).
        '''
        best_move = (None, None)
        best_score = self.game.LOSS if player == self.game.AI_PLAYER else self.game.WIN

        # terminal states
        if self.game.game_finished():
            best_score = self.game.board_state(self.game.AI_PLAYER)
            return (best_score, best_move)

        # go through all legal moves
        for cur_move in self.game.get_legal_moves():

            # make that move
            self.game.board[cur_move[0]][cur_move[1]] = player

            # maximizing player
            if player == self.game.AI_PLAYER:
                score = self.minimax(self.game.HUMAN_PLAYER,
                                     depth + 1, alpha, beta)[0]

                if best_score < score:
                    best_score = score - 10*depth
                    best_move = cur_move

                    alpha = max(alpha, best_score)
                    self.game.board[cur_move[0]][cur_move[1]] = self.game.EMPTY
                    if beta <= alpha:
                        break

            # minimizing player
            else:
                score = self.minimax(self.game.AI_PLAYER,
                                     depth + 1, alpha, beta)[0]

                if best_score > score:
                    best_score = score + 10*depth
                    best_move = cur_move

                    beta = min(beta, best_score)
                    self.game.board[cur_move[0]][cur_move[1]] = self.game.EMPTY
                    if beta <= alpha:
                        break

            # revert the move
            self.game.board[cur_move[0]][cur_move[1]] = self.game.EMPTY

        return (best_score, best_move)


if __name__ == '__main__':
    play_game(TicTacToe, agent=AlphaBetaPlayer)
