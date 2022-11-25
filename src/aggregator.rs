use std::marker::PhantomData;

use numpy::Ix2;
use numpy::ndarray::{Array, Ix1};

use crate::prelude::*;


pub trait Aggregator<S>: Send + Sync
where S: StrategyType
{
    fn n(&self) -> usize;
    fn u_i(&self, i: usize, strategies: &S) -> f64;
    fn u(&self, strategies: &S) -> Array<f64, Ix1> {
        Array::from_iter((0..strategies.n()).map(|i| self.u_i(i, strategies)))
    }
}


pub trait Discounter {
    fn gammas(&self) -> &Array<f64, Ix1>;
}

impl<S, T> Aggregator<S> for T
where S: StrategyType, T: StateIterator<St = dyn State<PFunc = dyn PayoffFunc<Act = S::Act>>> + Discounter
{
    fn n(&self) -> usize {
        self.state0().n()
    }
    fn u_i(&self, i: usize, strategies: &S) -> f64 {
        let actions_seq = strategies.actions();
        let state = &mut self.state0().clone();
        let gammas = self.gammas();
        let mut u = 0.0;
        for (t, actions) in actions_seq.iter().enumerate() {
            u += gammas[i].powi(t.try_into().unwrap()) * state.belief(i).u_i(i, actions);
            if t != strategies.t() - 1 {
                self.advance_state(state, actions);
            }
        }
        u
    }
    fn u(&self, strategies: &S) -> Array<f64, Ix1> {
        let actions_seq = strategies.actions();
        let state = &mut self.state0().clone();
        let gammas = self.gammas();
        let mut u: Array<f64, Ix1> = Array::zeros(gammas.len());
        for (t, actions) in actions_seq.iter().enumerate() {
            u.iter_mut().zip(gammas.iter()).enumerate().for_each(|(i, (u_i, gamma))| {
                *u_i += gamma.powi(t.try_into().unwrap()) * state.belief(i).u_i(i, actions);
            });
            if t != strategies.t() - 1 {
                self.advance_state(state, actions);
            }
        }
        u
    }
}


#[derive(Clone)]
pub struct FixedStateDiscounter<S, T>
where S: StrategyType, T: State<PFunc = dyn PayoffFunc<Act = S::Act>>
{
    pub state: T,
    pub gammas: Array<f64, Ix1>,
    _phantom: PhantomData<S>
}

impl<S, T> FixedStateDiscounter<S, T>
where S: StrategyType, T: State<PFunc = dyn PayoffFunc<Act = S::Act>>
{
    pub fn new(state: T, gammas: Array<f64, Ix1>) -> Result<Self, &'static str> {
        if state.n() != gammas.len() {
            return Err("When creating new FixedStateDiscounter: gammas must have length == n");
        }
        Ok(FixedStateDiscounter { state, gammas, _phantom: PhantomData })
    }
}

impl<S, T> StateIterator for FixedStateDiscounter<S, T>
where S: StrategyType + Clone, T: State<PFunc = dyn PayoffFunc<Act = S::Act>> + Clone
{
    type St = T;
    fn state0(&self) -> &T {
        &self.state
    }
}

impl<S, T> Discounter for FixedStateDiscounter<S, T>
where S: StrategyType, T: State<PFunc = dyn PayoffFunc<Act = S::Act>>
{
    fn gammas(&self) -> &Array<f64, Ix1> {
        &self.gammas
    }
}


#[derive(Clone)]
pub struct DynStateDiscounter<S, T>
where S: StrategyType,
      T: State<PFunc = dyn PayoffFunc<Act = S::Act>> + MutatesOn<S::Act>,
{
    pub state0: T,
    pub gammas: Array<f64, Ix1>,
    _phantom: PhantomData<S>,
}

impl<S, T> DynStateDiscounter<S, T>
where S: StrategyType,
      T: State<PFunc = dyn PayoffFunc<Act = S::Act>> + MutatesOn<S::Act>,
{
    pub fn new(state0: T, gammas: Array<f64, Ix1>) -> Result<Self, &'static str> {
        if state0.n() != gammas.len() {
            return Err("When creating new DynStateDiscounter: gammas must have length == n");
        }
        Ok(DynStateDiscounter { state0, gammas, _phantom: PhantomData })
    }
}

impl<S, T> StateIterator for DynStateDiscounter<S, T>
where S: StrategyType + Clone,
      T: State<PFunc = dyn PayoffFunc<Act = S::Act>> + MutatesOn<S::Act> + Clone,
{
    type St = T;
    fn state0(&self) -> &T {
        &self.state0
    }

    fn advance_state(&self, state: &mut T, actions: &S::Act) {
        state.mutate_on(actions);
    }
}

impl<S, T> Discounter for DynStateDiscounter<S, T>
where S: StrategyType,
      T: State<PFunc = dyn PayoffFunc<Act = S::Act>> + MutatesOn<S::Act> + Clone,
{
    fn gammas(&self) -> &Array<f64, Ix1> {
        &self.gammas
    }
}


#[derive(Clone)]
pub struct EndsOnContestWin<S, T, C>
where S: StrategyType + Clone,
      T: State<PFunc = ModularPayoff<S::Act>>,
      C: Discounter + StateIterator<St = T>
{
    pub child: C,
    _phantoms: PhantomData<(S, T)>,
}

impl<S, T, C> EndsOnContestWin<S, T, C>
where S: StrategyType + Clone,
      S::Act: Clone,
      T: State<PFunc = ModularPayoff<S::Act>> + MutatesOn<S::Act> + Clone,
      C: Discounter + StateIterator<St = T> + Clone
{
    pub fn new(child: C) -> Self {
        EndsOnContestWin { child, _phantoms: PhantomData }
    }
    pub fn probas(&self, strategies: &S) -> Array<f64, Ix2> {
        let mut probas = vec![1.; self.n()];
        let mut all_probas: Vec<f64> = Vec::with_capacity(self.n() * strategies.t());
        let actions_seq = strategies.actions();
        let mut state = self.child.state0().clone();
        for (t, actions) in actions_seq.iter().enumerate() {
            for i in 0..self.n() {
                all_probas.push(probas[i]);
                if t != strategies.t() - 1 {
                    let payoff_func = state.belief(i);
                    let (_, p) = payoff_func.prod_func.f(actions);
                    probas[i] *= 1. - payoff_func.csf.q(p.view()).iter().sum::<f64>();
                }
            }
            if t != strategies.t() - 1 {
                state.mutate_on(actions);
            }
        }
        Array::from_shape_vec((strategies.t(), self.n()), all_probas).unwrap()
    }
}

impl<S, T, C> StateIterator for EndsOnContestWin<S, T, C>
where S: StrategyType + Clone,
      S::Act: Clone,
      T: State<PFunc = ModularPayoff<S::Act>> + MutatesOn<S::Act> + Clone,
      C: Discounter + StateIterator<St = T> + Clone
{
    type St = <C as StateIterator>::St;
    fn state0(&self) -> &Self::St {
        self.child.state0()
    }

    fn advance_state(&self, state: &mut Self::St, actions: &S::Act) {
        state.mutate_on(actions);
    }
}

impl<S, T, C> Aggregator<S> for EndsOnContestWin<S, T, C>
where S: StrategyType + Clone,
      S::Act: Clone,
      T: State<PFunc = ModularPayoff<S::Act>> + MutatesOn<S::Act> + Clone,
      C: Discounter + StateIterator<St = T> + Clone
{
    fn n(&self) -> usize {
        self.child.n()
    }
    fn u_i(&self, i: usize, strategies: &S) -> f64 {
        let actions_seq = strategies.actions();
        let mut state = self.child.state0().clone();
        let gamma = self.child.gammas()[i];
        let mut proba = 1.;  // probability that nobody has won yet
        let mut u = 0.;
        for (t, actions) in actions_seq.iter().enumerate() {
            let payoff_func = state.belief(i);
            u += proba * gamma.powi(t.try_into().unwrap()) * payoff_func.u_i(i, actions);
            if t != strategies.t() - 1 {
                // update proba
                let (_, p) = payoff_func.prod_func.f(actions);
                proba *= 1. - payoff_func.csf.q(p.view()).iter().sum::<f64>();
                // update state
                self.advance_state(&mut state, actions);
            }
        }
        u
    }
    fn u(&self, strategies: &S) -> Array<f64, Ix1> {
        let actions_seq = strategies.actions();
        let state = &mut self.state0().clone();
        let gammas = self.child.gammas();
        let mut probas = vec![1.; gammas.len()];
        let mut u: Array<f64, Ix1> = Array::zeros(gammas.len());
        for (t, actions) in actions_seq.iter().enumerate() {
            u.iter_mut().zip(gammas.iter()).enumerate().for_each(|(i, (u_i, gamma))| {
                let payoff_func = state.belief(i);
                // update u
                *u_i += probas[i] * gamma.powi(t.try_into().unwrap()) * payoff_func.u_i(i, actions);
                if t != strategies.t() - 1 {
                    // update probas
                    let (_, p) = payoff_func.prod_func.f(actions);
                    probas[i] *= 1. - payoff_func.csf.q(p.view()).iter().sum::<f64>();
                }
            });
            if t != strategies.t() - 1 {
                // update state
                self.advance_state(state, actions);
            }
        }
        u
    }
}


pub type ExponentialDiscounter<T> = FixedStateDiscounter<Strategies<Actions>, T>;
pub type InvestExpDiscounter<T> = DynStateDiscounter<Strategies<InvestActions>, T>;
