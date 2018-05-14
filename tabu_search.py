# Author: Quentin Meeus

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import pickle
from collections import deque


class TabuSearchAlgorithm:

    COLUMN_NAMES = [
        'LotNr', 'Allocate', 'Bank', 'Dealer', 'Liquidator', 'Volunteer',
        'LotsSale', 'LotsCtgry', 'Forced', 'lEstValue', 'lFollowers',
        'Duration', 'Morning', 'Evening', 'Afternoon', 'lSP.EV'
    ]

    MOVE_NAMES = ["edit_dayperiod", "edit_duration", "edit_price"]
    PROBABILITIES = [.01, .49, .50]

    def __init__(self, initial_state, model, n_iter=10**3.5, early_stop=15, tolerance=.1,
                 tabu_size=20, verbose=True):


        # Algorithm configuration
        self.n_iter = n_iter
        self.early_stop = early_stop
        self.tolerance = tolerance
        self.tabu_size = tabu_size
        self.verbose = verbose

        # Algorithm initialisation
        self.iteration = 0
        self.model = self.load_model(model)
        self.state = initial_state
        self.score, self.predicted_proba = self.evaluate(initial_state)
        self.best = (self.score, self.state, self.predicted_proba)
        self.early_stop_history = deque(maxlen=early_stop)
        self.probabilities = {m: p for m, p in zip(self.MOVE_NAMES, self.PROBABILITIES)}
        self.move_functions = [self.edit_dayperiod, self.edit_duration, self.edit_price]
        self.tabu = {move: deque(maxlen=tabu_size) for move in self.MOVE_NAMES}

    def solve(self):
        """
        Perform the optimisation
        """
        while self.keep_running():

            # Random selection of a move
            func, params = self.choose_move()

            if not params:
                # Iteration is not incremented if no param was found
                continue

            self.iteration += 1

            # Perform the move
            self.move(func, params)

        # We make sure that all routes start at zero and end at zero
        self.print_report()

    def print_report(self):
        if self.verbose:
            print(f"Best solution achieved: {self.best[0]}\n"
                  f"Best probabilities achieved: {self.best[-1]}\n"
                  f"Best parameters found: \n{self.get_params(self.best[1])}")
            if self.max_iteration_reached():
                reason_stopped = "the maximum number of iteration was reached."
            elif self.early_stop_reached():
                reason_stopped = "the early stop criteria was met."
            else:
                reason_stopped = "I could not find satisfying parameters."
            print("I stopped because", reason_stopped)


    ############################################################################
    #                           OPERATIONS ON OBJECT
    ############################################################################

    def update(self, state, score, probabilities):
        """
        We update here the states if they are accepted by the method check (below)
        We also keep track of the best state that we encountered up to now and update
        the early_stop_history of the distances.
        :param state: the new state
        """
        if score > self.best[0]:
            self.best = (score, state, probabilities)
        self.early_stop_history.append(score)
        self.state = state
        self.score = score
        self.predicted_proba = probabilities
        
    def evaluate(self, row):
        probabilities = self.model.predict_proba(row.T.values.reshape(1, -1))[0]
        weighs = np.array([-1, 1, 5])
        score = np.dot(weighs, probabilities)
        return score, probabilities

    def get_params(self, row):
        print(row)
        return row[self.COLUMN_NAMES[-5:]]

    ############################################################################
    #                                   MOVES
    ############################################################################

    def move(self, func, params):
        """
        Perform one move, which is either a city swap (inside a route or between routes) or a two-edges
        exchange inside a route.
        :param func: the move function
        :param params: the parameters for the move function
        """
        new_state = func(self.state, params)
        new_score, probabilities = self.evaluate(new_state)

        # We check whether the new state is accepted or not and perform the update if it is
        if self.check(new_score):
            if self.verbose:
                print(
                    f"Score: {new_score:.2f}\tChange: {new_score / self.score - 1:+.3%} "
                    f"{func.__name__}, {params}"
                )
            self.update(new_state, new_score, probabilities)

    def choose_move(self):
        """
        This function choose a move to perform at each iteration with randomly selected parameters
        With each move is associated a probability, which becomes zero if the method generate_params
        cannot new generate parameters anymore.
        :return: the selected move function with random parameters
        """
        move_functions = self.move_functions
        choices = []
        for func_name, proba in self.probabilities.items():
            choices += [func_name] * int(proba * 100)
        choice = np.random.choice(choices)
        func = move_functions[list(map(lambda f: f.__name__, move_functions)).index(choice)]
        return func, self.generate_params(choice)

    def generate_params(self, func_name):
        """
        Generate random params for a specific move
        :param func_name: the name of the move (function) for which params should be generated
        :return: the parameters generated
        """
        param_functions = {
            "edit_dayperiod": self._params_dayperiod,
            "edit_duration": self._params_duration,
            "edit_price": self._params_price
        }
        tries = 0
        while True:
            if tries >= 10:
                return
            params = param_functions[func_name]()
            if params in self.tabu[func_name]:
                tries += 1
                continue
            self.tabu[func_name].append(params)
            return params

    @staticmethod
    def edit_dayperiod(row, value):
        row = row.copy()
        dayperiods = ["Morning", "Afternoon", "Evening"]
        new_period = dayperiods[value]
        for period in dayperiods:
            row[period] = 1 if period == new_period else 0
        return row

    @staticmethod
    def _params_dayperiod():
        return np.random.randint(0, 3)

    @staticmethod
    def edit_duration(row, value):
        row = row.copy()
        row["Duration"] = value
        return row

    @staticmethod
    def _params_duration():
        return np.random.randint(50, 1001)

    @staticmethod
    def edit_price(row, value):
        row = row.copy()
        # Set starting price to estimated price ration    
        row['lSP.EV'] = value
        return row

    def _params_price(self):
        return np.random.uniform(-1.5, 0.)

    ############################################################################
    #                                 CONDITIONS
    ############################################################################

    def check(self, score):
        """
        Two conditions must be met for a state to be accepted:
        1)  The new distance is less than the current distance * some threshold depending on the tolerance
            This allows for small upward moves in order not to be stuck in local optima.
        2)  The state must meet the distance constraint for a route to be feasible
        :param state: the new state
        :return: a boolean
        """
        current_score = self.score
        return score > current_score * (1 + self.tolerance)

    def keep_running(self):
        """
        One of three condition suffices to stop the algorithm:
        1)  All move functions have probabilities set to zero, i.e. we cannot find any new parameters
        2)  The max number of iteration has been reached
        3)  The conditions are met for the early stop
        :return: a boolean
        """
        return (any(proba > 0 for proba in self.probabilities.values())
                and self.best[0] < 4.5
                and not self.max_iteration_reached()
                and not self.early_stop_reached())

    def max_iteration_reached(self):
        return self.iteration >= self.n_iter

    def early_stop_reached(self):
        return self.best[0] not in self.early_stop_history and len(self.early_stop_history) == self.early_stop

    @staticmethod
    def load_model(model_file):
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        return model


if __name__ == '__main__':
    data2014 = pd.read_csv('data/data2014.csv').assign(year=2014)
    data2015 = pd.read_csv('data/data2015.csv').assign(year=2015)
    data = pd.concat([data2014, data2015], axis=0).reset_index(drop=True)

    target_name = "lmultiplier"
    train_cols = ['LotNr', 'Allocate', 'Bank', 'Dealer', 'Liquidator', 'Volunteer',
                  'LotsSale', 'LotsCtgry', 'Forced', 'lEstValue', 'lFollowers',
                  'Duration', 'Morning', 'Evening', 'Afternoon', 'lSP.EV']
    log_cols = ["multiplier", "EstValue", "StartPrice", "SP.EV", "Followers"]
    log10 = pd.DataFrame(np.log10(data[log_cols].values), columns=list(map("l{}".format, log_cols)))
    data = pd.concat([data, log10], axis=1).drop(log_cols, axis=1)

    sample = data.sample(1)

    X, y = sample[train_cols], sample["lmultiplier"]

    ts = TabuSearchAlgorithm(X, "output/AdaBoostClassifier.pkl", verbose=False)
    ts.solve()
    print("Final results:", 10**y.values[0], ts.model.predict(ts.best[1])[0], ts.best[0])
