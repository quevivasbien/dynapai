use numpy::ndarray::{Array, ArrayView, Ix2, s};
use argmin::core::{CostFunction, Executor};
use argmin::solver::neldermead::NelderMead;
use rayon::prelude::*;

use crate::prelude::*;

const INIT_MU: f64 = -1.;
const INIT_SIGMA: f64 = 0.1;


#[derive(Clone)]
pub enum InitGuess<A: ActionType> {
    Random(usize),
    Fixed(Strategies<A>),
}

impl<A: ActionType + Clone> InitGuess<A> {
    fn to_fixed(&self, n: usize) -> Strategies<A> {
        match self {
            InitGuess::Random(t) => Strategies::<A>::random(*t, n, INIT_MU, INIT_SIGMA).unwrap(),
            InitGuess::Fixed(x) => x.clone(),
        }
    }
}

#[derive(Clone)]
pub struct SolverOptions<A: ActionType + Clone> {
    pub init_guess: InitGuess<A>,
    pub max_iters: u64,
    pub tol: f64,
    pub nm_options: NMOptions,
}

impl<A: ActionType + Clone> SolverOptions<A> {
    pub fn from_init_guess(init_guess: Strategies<A>) -> Self {
        SolverOptions {
            init_guess: InitGuess::Fixed(init_guess),
            max_iters: 200,
            tol: 1e-6,
            nm_options: NMOptions::default(),
        }
    }

    pub fn random_init(t: usize) -> Self {
        SolverOptions {
            init_guess: InitGuess::Random(t),
            max_iters: 200,
            tol: 1e-6,
            nm_options: NMOptions::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct NMOptions {
    pub init_simplex_size: f64,
    pub max_iters: u64,
    pub tol: f64,
}

impl Default for NMOptions {
    fn default() -> Self {
        NMOptions {
            init_simplex_size: 0.1,
            max_iters: 200,
            tol: 1e-8,
        }
    }
}

struct PlayerObjective<'a, A: ActionType + Clone>{
    pub payoff_aggregator: &'a dyn Aggregator<A>,
    pub i: usize,
    pub base_strategies: &'a Strategies<A>,
}

// implement traits needed for argmin

impl<A: ActionType + Clone> CostFunction for PlayerObjective<'_, A> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let mut strategies = self.base_strategies.clone();
        strategies.set_i(
            self.i,
            Array::from_shape_vec(
                (self.base_strategies.t(), A::nparams()),
                params.iter().map(|x| x.exp()).collect(),
            )?
        );
        Ok(-self.payoff_aggregator.u_i(self.i, &strategies))
    }
}

fn create_simplex(init_guess: ArrayView<f64, Ix2>, init_simplex_size: f64) -> Vec<Vec<f64>> {
    let mut simplex = Vec::new();
    let base: Vec<f64> = init_guess.iter().map(|x| x.ln()).collect();
    for i in 0..base.len() {
        let mut x = base.clone();
        x[i] += init_simplex_size;
        simplex.push(x);
    }
    simplex.push(base);
    simplex
}

fn solve_for_i<A>(i: usize, strat: &Strategies<A>, agg: &dyn Aggregator<A>, options: &NMOptions) -> Result<Array<f64, Ix2>, argmin::core::Error>
where A: ActionType + Clone
{
    let init_simplex = create_simplex(
        strat.data().slice(s![.., i, ..]),
        options.init_simplex_size
    );
    let obj = PlayerObjective {
        payoff_aggregator: agg,
        i,
        base_strategies: strat,
    };
    let solver = NelderMead::new(init_simplex).with_sd_tolerance(options.tol)?;
    let res = Executor::new(obj, solver)
        .configure(|state| state.max_iters(options.max_iters))
        .run()?;
    Ok(Array::from_shape_vec(
        (strat.t(), <A as ActionType>::nparams()),
        res.state.best_param.unwrap().iter().map(|x| x.exp()).collect(),
    )?)
}

fn update_strat<A>(strat: &mut Strategies<A>, agg: &dyn Aggregator<A>, nm_options: &NMOptions) -> Result<(), argmin::core::Error>
where A: ActionType + Clone
{
    let new_data = (0..strat.n()).into_par_iter().map(|i| {
        solve_for_i(i, strat, agg, nm_options)
    }).collect::<Result<Vec<_>,_>>()?;
    for (i, x) in new_data.into_iter().enumerate() {
        strat.set_i(i, x);
    }
    Ok(())
}

fn within_tol<A: ActionType>(current: &Strategies<A>, last: &Strategies<A>, tol: f64) -> bool {
    isapprox_iters(
        current.data().iter().map(|x| x.ln()),
        last.data().iter().map(|x| x.ln()),
        tol, f64::EPSILON.sqrt()
    )
}

pub fn solve<A>(agg: &dyn Aggregator<A>, options: &SolverOptions<A>) -> Result<Strategies<A>, argmin::core::Error>
where A: ActionType + Clone
{
    let mut current_strat = options.init_guess.to_fixed(agg.n());
    for i in 0..options.max_iters {
        let last_strat = current_strat.clone();
        update_strat(&mut current_strat, agg, &options.nm_options)?;
        if within_tol(&current_strat, &last_strat, options.tol) {
            println!("Exited on iteration {}", i);
            return Ok(current_strat);
        }
    }
    println!("Reached max iterations ({})", options.max_iters);
    Ok(current_strat)
}
