import numpy as np
import dynapai as dp

actions = dp.Actions.from_inputs(xs = np.array([1., 1.]), xp = np.array([2., 2.]))
print(f'actions = {actions}')

investActions = dp.InvestActions.from_inputs(
    xs = np.array([1., 1.]), xp = np.array([2., 2.]),
    inv_s = np.array([3., 3.]), inv_p = np.array([4., 4.])
)
print(f'investActions = {investActions}')

prodFunc = dp.ProdFunc(
    a = np.array([10., 10.]),
    alpha = np.array([0.5, 0.5]),
    b = np.array([10., 10.]),
    beta = np.array([0.5, 0.5]),
)
print(f'prodFunc = {prodFunc}')

product = prodFunc(actions)
print(f'product of actions = {product}')

invest_product = prodFunc(investActions)
print(f'product of investActions = {invest_product}')


payoffFunc = dp.PayoffFunc(
    prod_func = prodFunc,
    theta = np.array([0.5, 0.5]),
    d = np.array([1., 1.]),
    r = np.array([0.1, 0.1]),
)
print(f'payoffFunc = {payoffFunc}')

payoff = payoffFunc(actions)
print(f'payoff of actions = {payoff}')

investPayoffFunc = dp.PayoffFunc(
    prod_func = prodFunc,
    theta = np.array([0.5, 0.5]),
    d = np.array([1., 1.]),
    r = np.array([0.1, 0.1]),
    r_inv = np.array([0.2, 0.2]),
)
print(f'investPayoffFunc = {investPayoffFunc}')

invest_payoff = investPayoffFunc(investActions)
print(f'payoff of investActions = {invest_payoff}')

agg = dp.Aggregator(
    state = payoffFunc,
    gammas = np.array([0.95, 0.9]),
    end_on_win = True,
)

print(f'multi-period payoff = {agg.u([actions, actions])}')

sol = agg.solve(init = 2);
print(f'solution:', '\n'.join(str(x) for x in sol), sep = '\n')

invest_agg = dp.Aggregator(
    state = investPayoffFunc,
    gammas = np.array([0.95, 0.9]),
    end_on_win = True,
)

print(f'multi-period invest payoff = {invest_agg.u([investActions, investActions])}')

invest_sol = invest_agg.solve(init = 2);
print(f'invest solution:', '\n'.join(str(x) for x in invest_sol), sep = '\n')
