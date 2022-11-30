import argparse
import numpy as np
import dynapai as dp

from time import time

class Tester:
    def __init__(self, n, t):
        self.n = n
        self.t = t
        
        self.gammas = np.linspace(0.1, 0.9, n)

        self.prodFunc = dp.ProdFunc(
            a = np.full(n, 10.),
            alpha = np.full(n, 0.5),
            b = np.full(n, 1.),
            beta = np.full(n, 0.5),
        )
        self.riskFunc = dp.RiskFunc.winner_only_risk(np.full(n, 0.5))
    
    def solve_agg(self, agg, strat_type = 'strategies', plot = False):
        print(f"Solving for optimal {strat_type}, with {self.n} players and {self.t} time steps...")
        time0 = time()
        res = agg.solve(self.t)
        time1 = time()
        print(f"Solved in {time1 - time0:.3f} seconds")
        print(f"Optimal {strat_type}:\n{res}")
        print(f"Payoff from optimal {strat_type}: {agg.u(res.optimum())}")
        print()

        if plot:
            dp.plot(res, title = f"Optimal {strat_type}")
        return res
    
    def get_basic_agg(self, end_on_win = False):
        payoffFunc = dp.PayoffFunc(
            prod_func = self.prodFunc,
            reward_func = dp.RewardFunc(self.n),
            csf = dp.CSF.default(),
            risk_func = self.riskFunc,
            d = np.full(self.n, 1.),
            r = np.full(self.n, 0.1),
        )

        return dp.Aggregator(
            state = payoffFunc,
            gammas = self.gammas,
            end_on_win = end_on_win,
        )

    def solve_basic(self, plot = False):
        agg = self.get_basic_agg()
        return self.solve_agg(agg, plot = plot)

    def get_invest_agg(self):
        payoffFunc = dp.PayoffFunc(
            prod_func = self.prodFunc,
            reward_func = dp.RewardFunc(self.n),
            csf = dp.CSF.maybe_no_win(),
            risk_func = self.riskFunc,
            d = np.full(self.n, 1.),
            r = np.full(self.n, 0.1),
            r_inv = np.full(self.n, 0.01),
        )

        return dp.Aggregator(
            state = payoffFunc,
            gammas = self.gammas
        )

    def solve_invest(self, plot = False):
        agg = self.get_invest_agg()
        return self.solve_agg(agg, strat_type = 'invest strategies', plot = plot)
    
    def solve_end_on_win(self, plot = False):
        child = self.get_basic_agg(end_on_win = True)
        agg = dp.Aggregator(child)
        res = self.solve_agg(agg, strat_type = 'strategies (end-on-win)', plot = plot)
        print("Players' beliefs about probabilities of reaching each t:", agg.probas(res), sep = '\n')
        return res
    
    def solve_invest_end_on_win(self, plot = False):
        agg = self.get_invest_agg(end_on_win = True)
        res = self.solve_agg(agg, strat_type = 'invest strategies (end-on-win)', plot = plot)
        print("Players' beliefs about probabilities of reaching each t:", agg.probas(res), sep = '\n')
        return res

    def solve_scenario(self):
        # create two prod funcs with different values of theta
        payoff_funcs = dp.PayoffFunc.expand_from(
            prod_func_list = [self.prodFunc],
            risk_func_list = [
                dp.RiskFunc.winner_only_risk(np.full(self.n, 0.5)),
                dp.RiskFunc.winner_only_risk(np.full(self.n, 1.0))
            ],
            csf_list = [dp.CSF.maybe_no_win()],
            reward_func_list = [dp.RewardFunc(self.n)],
            d_list = [np.full(self.n, 1.)],
            r_list = [np.full(self.n, 0.1)],
            r_inv_list = [np.full(self.n, 0.01)],
        )

        aggs = dp.Aggregator.expand_from(
            state_list = payoff_funcs,
            gammas_list = [self.gammas],
        )

        print("Trying [parallel] solve of scenario...")
        scenario = dp.Scenario(aggs)
        time0 = time()
        res = scenario.solve(self.t)
        time1 = time()
        print(f"Solved in {time1 - time0:.3f} seconds")
        print("Optimal invest strategies:")
        for i, r in enumerate(res):
            print(f'Problem {i+1}:\n{r}\n')




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type = int, default = 2, help = 'number of players in test scenarios')
    parser.add_argument('--t', type = int, default = 5, help = 'number of time steps in test scenarios')
    parser.add_argument('--basic', action = 'store_true', help = 'solve basic problem')
    parser.add_argument('--invest', action = 'store_true', help = 'solve problem with investment')
    parser.add_argument('--end-on-win', action = 'store_true', help = 'solve basic problem with end-on-win condition')
    parser.add_argument('--invest-end-on-win', action = 'store_true', help = 'solve problem with investment and end-on-win condition')
    parser.add_argument('--scenario', action = 'store_true', help = 'solve multiple invest problems in parallel')
    parser.add_argument('--all', action = 'store_true', help = 'run all tests')
    parser.add_argument('--plot', action = 'store_true', help = 'plot results')

    args = parser.parse_args()
    tester = Tester(args.n, args.t)
    if args.basic or args.all:
        tester.solve_basic(args.plot)
    if args.invest or args.all:
        tester.solve_invest(args.plot)
    if args.end_on_win or args.all:
        tester.solve_end_on_win(args.plot)
    if args.invest_end_on_win or args.all:
        tester.solve_invest_end_on_win(args.plot)
    if args.scenario or args.all:
        tester.solve_scenario()

if __name__ == '__main__':
    main()
