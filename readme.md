# dynapai

## What is this?

This repository contains code meant to model interactions between actors competing to develop a new, risky technology (we have AI technology in mind in particular, but this could be any technology that carries some risk of unintended negative consequences).

A lot of the functionality of the code here overlaps with [AIIncentives.jl](https://github.com/quevivasbien/AIIncentives.jl/). Both this code and that code are based on the same basic model, though this code specializes in multi-period extensions of that model, while AIIncentives.jl is meant to provide a robust set of tools for studying the static model.

## Getting started

The easiest way to use this package is likely via the provided Python bindings. I provide Python wheels in the releases section; if one of those wheels is compatible with your OS and Python version, you should be able to download it and install it with, e.g., pip.

### How to build?

If you want to interact directly with the Rust code or build for an unsupported version of Python, you'll need to take the following steps (otherwise, just skip this section)

You'll probably want to start by creating a new virtual environment: if you have Conda (e.g., Anaconda), you can do that by running
```bash
conda create --name pyo3-dev python=3.7
```
You can create your venv in some other way and call it whatever you want, but you do need the Python version to be at least 3.7.

You'll then need to activate the virtual environment. With Conda:
```bash
conda activate pyo3-dev
```

You can then install the maturin build tool in this environment with
```bash
pip install maturin
```

Finally, to compile the Rust code and create a Python library, run the following from the main directory of this repository:
```bash
maturin develop
```

(To get added performance, run with the `--release` option, i.e., `maturin develop --release`.)

The `maturin develop` command builds the library and installs it on the active Python environment (the venv you created). If you instead want to build a wheel to install elsewhere, you'll need to use the `maturin build` command.

As long as you have the venv you created active, you should then be able to import the Python bindings in a module called `dynapai`.


## How to use

Once installed, you can use this like any other Python package. Here's a simple example:

```python
import dynapai as dp

prod_func = dp.ProdFunc(
    a = [10., 10.],
    alpha = [0.5, 0.5],
    b = [10., 10.],
    beta = [0.5, 0.5]
)

payoff_func = dp.PayoffFunc(
    prod_func = prod_func,
    risk_func = dp.RiskFunc.winner_only([0.5, 0.5]),
    d = [1.0, 1.0],
    r = [0.1, 0.1]
)

actions = dp.Actions.from_inputs([1., 1.], [2., 2.])

print(f"Payoff from actions: {payoff_func(actions)}")
```

This should print
```
Payoff from actions: [-0.2099315 -0.2099315]
```

To make sure things are working correctly, you can try running the `demo.py` script in this directory. That script also gives a good idea of what features are available in the package.


## What does this model?

[AIIncentives.jl](https://github.com/quevivasbien/AIIncentives.jl/) models games where players choose $x_s, x_p$ and get payoffs of the form
$$u_i = \sum_j \sigma_j(s) q_j(p) \rho_{ij}(p) - \sum_j (1 - \sigma_j(s)) q_j(p) d_{ij} - c_i(x_s, x_p)$$
where
$$s = Ax_s^\alpha p^{-\theta}, \quad p = Bx_p^\beta.$$

The notation used is:

- $\sigma_j$ is the probability of a safe outcome given that $j$ wins the contest
- $q_j$ is the probability that $j$ wins the contest
- $\rho_{ij}$ is the payoff to player $i$ if $j$ wins the contest
- $d_{ij}$ is the cost of an unsafe (disaster) outcome to player $i$ if $j$ wins the contest
- $c_i$ is the cost paid to use inputs $x_s, x_p$

This package can model those same games, but it can also model dynamic versions of those games, where players choose a schedule $x_s(t), x_p(t)$ of strategies over some number of time periods. We can also assume that the model parameters vary over periods, possibly in response to players' actions.

In a typical case, we'll assume that players are exponential discounters; i.e., each player has some discount rate $\gamma_i$, and their total payoff over $T$ time periods (from the perspective of period 0) is
$$U_i = \sum_{t=1}^T \gamma_i^t u_i(t).$$

Each player chooses a strategy $x_i = \{x_{s,i}, x_{p,i}\}$, and a set $x^*$ of pure strategies is a Nash equilibrium
$$x_i^* = \left. \argmax_{x_i} U_i(x) \right|_{x_{-i} = x_{-i}^*}$$
for all $i$.

### Choice of $T$

The number of time periods ($T$) could be fixed, or it could depend on players' strategies.

One interesting case is to assume that the game ends the first time someone wins the contest. In this case, $T$ is a random variable -- the probability that someone wins in a given period $t$ is
$$\sum_i q_i(p(t))$$
so if the probabilities of someone winner are independent between periods, we have
$$\Pr\{T = t\} = 1 - \prod_{s=1}^t\left(1 - \sum_i q_i(p(s)) \right).$$
Each player's objective is
$$U_i = \sum_{t=1}^\infty \gamma_i^t \Pr\{T \leq t\} u_i(t).$$

<!-- might be able to use bellman eqn to solve for parameterized solution of infinite period case -->

We might also impose a cutoff time $T_{max}$, meaning the objective would be
$$U_i = \sum_{t=1}^{T_{max}} \gamma_i^t \Pr\{T \leq t\} u_i(t).$$

<!-- In the static model, we typically assume that
$$q_i(p) = \frac{p_i}{\sum_j p_j},$$
but that doesn't make much sense in this example, since it gives $\sum q_i = 1$.

A plausible assumption might have the form
$$q_i(p) = \Pr\{\text{player }i\text{ wins | someone wins}\} \Pr\{\text{someone wins}\},$$
e.g.,
$$q_i(p) = \frac{q_i}{Q} \cdot \frac{Q}{1 + Q}, \quad Q = \sum_j q_j$$

This is the form used by the `MaybeNoWinCSF` type defined in `csf.rs`. -->

### Allowing players to invest

In addition to players choosing the amounts of effort that they want to expend on safety and performance ($x_s$ and $x_p$, respectively), we might assume that players can also invest in their future productivity; in each period, we might let them choose quantities $i_s$ and $i_p$ to improve their productivity, and then let
$$A(t+1) = A(t) + i_s(t), \quad B(t+1) = B(t) + i_p(t).$$
A strategy is thus a set
$$x_i = \{x_{s,i}, x_{p,i}, i_{s,i}, i_{p,i}\}.$$

(This is the form used by the `InvestActions` type in the package.)

### Allowing players to share technology

If we want to model technology/knowledge spillovers and assume that players can choose how much information they share, we can introduce choice variables $ʃ_s$ and $ʃ_p$. We can then assume a variation of [who?]:
$$A_i(t+1) = A_i(t) + i_{s,i}(t) + \sum_{j \neq i} ʃ_{p,j}(t) \max\{0, A_j(t) - A_i(t)\}$$
$$B_i(t+1) = B_i(t) + i_{p,i}(t) + \sum_{j \neq i} ʃ_{p,j}(t) \max\{0, B_j(t) - B_i(t)\}$$

The interpretation here is that players may choose to share a fraction $ʃ$ of their technology lead with their competitors. In this context, a strategy takes the form
$$x_i = \{x_{s,i}, x_{p,i}, i_{s,i}, i_{p,i}, ʃ_{s,i}, ʃ_{p,i}\}.$$

(This is the form used by the `SharingActions` type in the package.)


## Package setup

This section is meant to give an overview of how the package code is arranged.

### Actions and strategies

The fundamental trait is the `ActionType`, which represents actions for *all players* in a single time period. There are three action types implemented by default:

* `Actions` -- represents $\{x_s(t), x_p(t)\}$ for some $t$
* `InvestActions` -- represents $\{x_s(t), x_p(t), i_s(t), i_p(t)\}$ for some $t$
* `SharingActions` -- represents $\{x_s(t), x_p(t), i_s(t), i_p(t), ʃ_s(t), ʃ_p(t)\}$ for some $t$

A vector of actions can be packaged into a `Strategies` object, the sequence of actions that constitute a set of strategies for all players.

Key idea: actions are for a single period (but all players); strategies are for multiple time periods (and all players).

All of the following types of objects come associated with a specific action type. That means that you cannot define, for example, a payoff function meant for `InvestActions` and then use it on `SharingActions` -- you need to make sure that all the model components you use are compatible with the type of actions you want to work with.

### Payoff function

An object with the `PayoffFunction` trait defines a utility function (payoff) for a single period -- i.e. it maps from actions to player payoffs.

The default payoff function is the `ModularPayoff` which calculates a payoff from the following components:

* A production function, which determines how actions/strategies translate into the outputs $s$ and $p$. The default implementation has $s = Ax_s^\alpha$ and $p = Bx_p^\beta$. When players make investments and/or share technology, the production function parameters are mutated between time periods.

* A risk function, which determines the probability of a disaster outcome conditional on each player winning the contest.

* A reward function, which determines a matrix of rewards, what player $i$ gets if player $j$ wins

* A contest success function (CSF), which determines the probability of winning for each player

* A disaster cost function, which determines a matrix of penalties, what player $i$ gets if player $j$ causes a disaster

* A cost function, which determines the price each player pays for their choice of actions

### State

An object with the `State` trait describes players beliefs about the payoff function in a given time period. A simple version is the `CommonBeliefs` type, which just encapsulates a single payoff function.

### Aggregator

An object with the `Aggregator` trait can take a starting state and a strategy set (`Strategies` object) and calculate (aggregate) players payoffs from those. The default implementations assume players discount future payoffs exponentially.
