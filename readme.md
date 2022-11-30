# ai-dynamic-game

## What is this?

This repository contains code meant to model interactions between actors competing to develop a new, risky technology (we have AI technology in mind in particular, but this could be any technology that carries some risk of unintended negative consequences).

A lot of the functionality of the code here overlaps with [AIIncentives.jl](https://github.com/quevivasbien/AIIncentives.jl/). Both this code and that code are based on the same basic model, though this code specializes in multi-period extensions of that model, while AIIncentives.jl is meant to provide a robust set of tools for studying the static model.

## How to build?

Currently, you need to be able to compile Rust to interact with this project, though I may provide a pre-built Python library in the future.

You can interact directly with the Rust code, or use the provided Python bindings to access some functionality in Python.

### Using the Python bindings

Start by creating a new virtual environment: if you have Conda (e.g., Anaconda), you can do that by running
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

((You can also compile a Python library wheel with `maturin build`, but I wouldn't recommend this until this project is in a more developed state.))

As long as you have the venv you created active, you should then be able to import the Python bindings in a module called `dynapai`; e.g., in Python:
```python
import dynapai as dp

prodFunc = dp.ProdFunc(
    a = [10., 10.],
    alpha = [0.5, 0.5],
    b = [10., 10.],
    beta = [0.5, 0.5]
)

payoffFunc = dp.PayoffFunc(
    prod_func = prodFunc,
    risk_func = dp.RiskFunc.winner_only_risk([0.5, 0.5]),
    d = [1.0, 1.0],
    r = [0.1, 0.1]
)

actions = dp.Actions.from_inputs([1., 1.], [2., 2.])

print(f"Payoff from actions: {payoffFunc(actions)}")
```

This should print
```
Payoff from actions: [-0.2099315 -0.2099315]
```

To make sure things are working correctly, you can try running the `test.py` script in this directory.


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

This crate can model those same games, but it can also model dynamic versions of those games, where players choose a schedule $x_s(t), x_p(t)$ of strategies over some number of time periods. We can also assume that the model parameters vary over periods, possibly in response to players' actions.

Here's an example game that may be of interest: We'll assume that $n$ actors play over $T$ time periods, where $T$ is the first time period in which one of the actors wins the competition. Note that $T$ is a random variable; the probability that someone wins in a given period $t$ is
$$\sum_i q_i(p(t)),$$
and assuming that the probabilities of someone winning are independent between periods,
$$\Pr\{T = t\} = 1 - \prod_{s=1}^t\left(1 - \sum_i q_i(p(s)) \right)$$

In the static model, we typically assume that
$$q_i(p) = \frac{p_i}{\sum_j p_j},$$
but that doesn't make much sense in this example, since it gives $\sum q_i = 1$.

A plausible assumption might have the form
$$q_i(p) = \Pr\{\text{player }i\text{ wins | someone wins}\} \Pr\{\text{someone wins}\},$$
e.g.,
$$q_i(p) = \frac{q_i}{Q} \cdot \frac{Q}{1 + Q}, \quad Q = \sum_j q_j$$

This is the form used by the `MaybeNoWinCSF` type defined in `csf.rs`.