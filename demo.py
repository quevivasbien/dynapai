import argparse
import numpy as np
import dynapai as dp
import sys

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
        self.rewardFunc = dp.RewardFunc(n)
        self.riskFunc = dp.RiskFunc.winner_only_risk(np.full(n, 0.5))
    
    def solve_agg(self, agg, strat_type = 'strategies', plot = False):
        print(f"Solving for optimal {strat_type}, with {self.n} players and {self.t} time steps...")
        time0 = time()
        res = agg.solve(self.t)
        time1 = time()
        print(f"Solved in {time1 - time0:.3f} seconds")
        print(f"Optimal {strat_type}:\n{res}")
        optimum = res.optimum()
        print(f"Payoff from optimal {strat_type}: {agg.u(optimum)}")
        print()

        if plot:
            dp.plot(optimum, title = f"Optimal {strat_type}")
        return res
    
    def get_basic_agg(self, end_on_win = False):
        payoffFunc = dp.PayoffFunc(
            prod_func = self.prodFunc,
            reward_func = self.rewardFunc,
            csf = dp.CSF.default(),
            risk_func = self.riskFunc,
            d = np.full(self.n, 1.),
            cost_func = dp.CostFunc.fixed(np.full(self.n, 0.1)),
        )

        return dp.Aggregator(
            state = payoffFunc,
            gammas = self.gammas,
            end_on_win = end_on_win,
        )

    def solve_basic(self, plot = False, end_on_win = False):
        agg = self.get_basic_agg(end_on_win)
        return self.solve_agg(agg, plot = plot)

    def get_invest_agg(self, end_on_win = False):
        payoffFunc = dp.PayoffFunc(
            prod_func = self.prodFunc,
            reward_func = self.rewardFunc,
            csf = dp.CSF.maybe_no_win(),
            risk_func = self.riskFunc,
            d = np.full(self.n, 1.),
            cost_func = dp.CostFunc.fixed_invest(
                np.full(self.n, 0.1),
                np.full(self.n, 0.01),
            ),
        )

        return dp.Aggregator(
            state = payoffFunc,
            gammas = self.gammas,
            end_on_win = end_on_win,
        )

    def solve_invest(self, plot = False, end_on_win = False):
        agg = self.get_invest_agg(end_on_win)
        return self.solve_agg(agg, strat_type = 'invest strategies', plot = plot)

    def solve_scenario(self):
        # create two prod funcs with different values of theta
        payoff_funcs = dp.PayoffFunc.expand_from(
            prod_func_list = [self.prodFunc],
            risk_func_list = [
                dp.RiskFunc.winner_only_risk(np.full(self.n, 0.5)),
                dp.RiskFunc.winner_only_risk(np.full(self.n, 1.0))
            ],
            csf_list = [dp.CSF.maybe_no_win()],
            reward_func_list = [self.rewardFunc],
            d_list = [np.full(self.n, 1.)],
            cost_func_list = [dp.CostFunc.fixed_invest(
                np.full(self.n, 0.1),
                np.full(self.n, 0.01),
            )],
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
    parser.add_argument('--scenario', action = 'store_true', help = 'solve multiple invest problems in parallel')
    parser.add_argument('--all', action = 'store_true', help = 'run all tests')
    parser.add_argument('--end-on-win', action = 'store_true', help = 'end game the first time someone wins')
    parser.add_argument('--plot', action = 'store_true', help = 'plot results')

    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    tester = Tester(args.n, args.t)
    if args.basic or args.all:
        tester.solve_basic(args.plot, args.end_on_win)
    if args.invest or args.all:
        tester.solve_invest(args.plot, args.end_on_win)
    if args.scenario or args.all:
        tester.solve_scenario()

if __name__ == '__main__':
    main()
