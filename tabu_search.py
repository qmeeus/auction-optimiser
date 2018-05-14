# Author: Quentin Meeus

from copy import deepcopy
import random
from collections import deque


class TabuSearchAlgorithm:

    MOVE_FUNCTIONS = ["local_city_swap", "two_edges_exchange", "global_city_swap"]
    PROBABILITIES = [.01, .49, .5]

    def __init__(self, n_iter=10**5, early_stop=300, tolerance=.0007, tabu_size=15, verbose=True):
        self.n_iter = n_iter
        self.early_stop = early_stop
        self.tolerance = tolerance
        self.tabu_size = tabu_size
        self.verbose = verbose

        self.iteration = 0
        self.current_best = (0, None)
        self.history = deque(maxlen=early_stop)
        self.probabilities = {move: proba for move, proba in zip(self.MOVE_FUNCTIONS, self.PROBABILITIES)}
        self.move_functions = [self.local_city_swap, self.two_edges_exchange, self.global_city_swap]
        self.tabu = {move: deque(maxlen=tabu_size) for move in self.MOVE_FUNCTIONS}

    def solve(self):
        """
        Perform the optimisation
        """
        while self.keep_running():

            # Random selection of a move
            func, params = self.choose_move()

            # If we could not find any params for the chosen move, we won't try it again
            if not params:
                input("Removing %s" % func.__name__)
                self.probabilities[func.__name__] = 0
                continue

            # Iteration is not incremented if no param was found
            self.iteration += 1

            # Perform the move
            self.move(func, params)

        # We make sure that all routes start at zero and end at zero
        self.solution_ = {vehicle: rotate(route) for vehicle, route in self.current_best[1].items()}
        self.print_report()

    def print_report(self):
        if self.verbose:
            print("Best solution achieved:", self.current_best[0])
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

    def update(self, state):
        """
        We update here the states if they are accepted by the method check (below)
        We also keep track of the best state that we encountered up to now and update
        the history of the distances.
        :param state: the new state
        """
        new_distance = self.objective_function(state)
        if new_distance < self.current_best[0]:
            self.current_best = (new_distance, state)
        self.history.append(new_distance)
        self.state = state
        
    def objective_function(self, row):
        current = self.proba
        proba = self.model.predict_proba(row)

        return

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
        # Calculate the new state
        new_state = func(*params)

        # We check whether the new state is accepted or not and perform the update if it is
        if self.check(new_state):
            if self.verbose:
                new_distance = self.objective_function(new_state)
                print("Distance: {:.2f}\tChange: {:+.3%}"
                      .format(new_distance, new_distance / self.objective_function(self.state) - 1))
            self.update(new_state)

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
        chosen_func_name = random.choice(choices)
        func = move_functions[list(map(lambda f: f.__name__, move_functions)).index(chosen_func_name)]
        return func, self.generate_params(func.__name__)

    def generate_params(self, func_name):
        """
        Generate random params for a specific move
        :param func_name: the name of the move (function) for which params should be generated
        :return: the parameters generated
        """
        param_functions = {
            "local_city_swap": self._params_local_city_swap,
            "two_edges_exchange": self._params_two_edges_exchange,
            "global_city_swap": self._params_global_city_swap
        }
        tries = 0
        while True:
            if tries >= self.move_invalidation_threshold:
                return
            params = param_functions[func_name]()
            if params in self.tabu[func_name]:
                tries += 1
                continue
            self.tabu[func_name].append(params)
            return params
        
    def set_daytime(self, lot):
        # Change auction daytime randomly only one time can be picked at the same time
        dayperiods = ["Morning", "Afternoon", "Evening"]
        random_period, = np.random.choice(dayperiods, 1)
        for period in dayperiods:
            lot[period] = 1 if period == random_period else 0
        return lot

    def set_duration(self, lot):
        # Change auction duration
        lot["Duration"] = np.random.randint(50,1001)
        return lot

    def set_start_price(self, lot):
        # Set starting price to estimated price ration    
        spev = np.random.random() * 1.6 + 0.07
        lot['lSP.EV'] = np.log10(spev)
        # Because we change the sp.ev we also adapt the startprice
        lot['lStartPrice'] = np.log10(spev * 10 ** lot['lEstValue'])
        return lot

    ############################################################################
    #                                 CONDITIONS
    ############################################################################

    def check(self, state):
        """
        Two conditions must be met for a state to be accepted:
        1)  The new distance is less than the current distance * some threshold depending on the tolerance
            This allows for small upward moves in order not to be stuck in local optima.
        2)  The state must meet the distance constraint for a route to be feasible
        :param state: the new state
        :return: a boolean
        """
        current_distance = self.objective_function(self.state)
        return (self.objective_function(state) < current_distance * (1 + self.tolerance)
                and self.check_distance_constraint(state))

    def keep_running(self):
        """
        One of three condition suffices to stop the algorithm:
        1)  All move functions have probabilities set to zero, i.e. we cannot find any new parameters
        2)  The max number of iteration has been reached
        3)  The conditions are met for the early stop
        :return: a boolean
        """
        return (any(proba > 0 for proba in self.probabilities.values())
                and not self.max_iteration_reached()
                and not self.early_stop_reached())

    def max_iteration_reached(self):
        return self.iteration >= self.n_iter

    def early_stop_reached(self):
        return self.current_best[0] not in self.history and len(self.history) == self.early_stop
