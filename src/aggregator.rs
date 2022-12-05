use downcast_rs::{Downcast, impl_downcast};
use dyn_clone::{DynClone, clone_trait_object};
use numpy::Ix2;
use numpy::ndarray::{Array, Ix1};

use crate::prelude::*;


pub trait Aggregator<A: ActionType>: StateIterator<A> + Downcast + DynClone + Send + Sync {
    fn n(&self) -> usize;
    fn u_i(&self, i: usize, strategies: &Strategies<A>) -> f64;
    fn u(&self, strategies: &Strategies<A>) -> Array<f64, Ix1> {
        Array::from_iter((0..strategies.n()).map(|i| self.u_i(i, strategies)))
    }
}

clone_trait_object!(<A> Aggregator<A> where A: ActionType);
impl_downcast!(Aggregator<A> where A: ActionType);


pub trait Discounter {
    fn gammas(&self) -> &Array<f64, Ix1>;
}

impl<A: ActionType + 'static, T: StateIterator<A> + Discounter + 'static> Aggregator<A> for T {
    fn n(&self) -> usize {
        self.state0().n()
    }
    fn u_i(&self, i: usize, strategies: &Strategies<A>) -> f64 {
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
    fn u(&self, strategies: &Strategies<A>) -> Array<f64, Ix1> {
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
pub struct FixedStateDiscounter<A: ActionType> {
    pub state: Box<dyn State<A>>,
    pub gammas: Array<f64, Ix1>,
}

impl<A: ActionType + 'static> FixedStateDiscounter<A> {
    pub fn new(state: Box<dyn State<A>>, gammas: Array<f64, Ix1>) -> Result<Self, &'static str> {
        if state.n() != gammas.len() {
            return Err("When creating new FixedStateDiscounter: gammas must have length == n");
        }
        Ok(FixedStateDiscounter { state, gammas })
    }
}

impl<A: ActionType + Clone> StateIterator<A> for FixedStateDiscounter<A> {
    fn state0(&self) -> &Box<dyn State<A>> {
        &self.state
    }
}

impl<A: ActionType> Discounter for FixedStateDiscounter<A> {
    fn gammas(&self) -> &Array<f64, Ix1> {
        &self.gammas
    }
}


#[derive(Clone)]
pub struct DynStateDiscounter<A: ActionType> {
    pub state0: Box<dyn State<A>>,
    pub gammas: Array<f64, Ix1>,
}

impl<A: ActionType + 'static> DynStateDiscounter<A> {
    pub fn new(state0: Box<dyn State<A>>, gammas: Array<f64, Ix1>) -> Result<Self, &'static str> {
        if state0.n() != gammas.len() {
            return Err("When creating new DynStateDiscounter: gammas must have length == n");
        }
        Ok(DynStateDiscounter { state0, gammas })
    }
}

impl<A: ActionType + Clone> StateIterator<A> for DynStateDiscounter<A>
{
    fn state0(&self) -> &Box<dyn State<A>> {
        &self.state0
    }

    fn advance_state(&self, state: &mut Box<dyn State<A>>, actions: &A) {
        state.mutate_on(actions);
    }
}

impl<A: ActionType> Discounter for DynStateDiscounter<A> {
    fn gammas(&self) -> &Array<f64, Ix1> {
        &self.gammas
    }
}


#[derive(Clone)]
pub struct EndsOnContestWin<A, C>
where A: ActionType,
      C: Discounter + StateIterator<A>
{
    pub child: C,
    _phantom: std::marker::PhantomData<A>,
}

impl<A, C> EndsOnContestWin<A, C>
where A: ActionType + Clone + 'static,
      C: Discounter + StateIterator<A> + Clone + 'static
{
    pub fn new(child: C) -> Result<Self, &'static str> {
        // check that the states given by the child have ModularPayoff beliefs
        let state0 = child.state0();
        for i in 0..state0.n() {
            if let None = state0.belief(i).downcast_ref::<ModularPayoff<A>>() {
                return Err("The provided states should all contain ModularPayoff types")
            }
        }
        Ok(EndsOnContestWin { child, _phantom: std::marker::PhantomData })
    }
    pub fn probas(&self, strategies: &Strategies<A>) -> Array<f64, Ix2> {
        let mut probas = vec![1.; self.n()];
        let mut all_probas: Vec<f64> = Vec::with_capacity(self.n() * strategies.t());
        let actions_seq = strategies.actions();
        let mut state = self.child.state0().clone();
        for (t, actions) in actions_seq.iter().enumerate() {
            for i in 0..self.n() {
                all_probas.push(probas[i]);
                if t != strategies.t() - 1 {
                    let payoff_func = state.belief(i).downcast_ref::<ModularPayoff<A>>().expect(
                        "Belief should be ModularPayoff, but found something else"
                    );
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

impl<A, C> StateIterator<A> for EndsOnContestWin<A, C>
where A: ActionType + Clone,
      C: Discounter + StateIterator<A> + Clone
{
    fn state0(&self) -> &Box<dyn State<A>> {
        self.child.state0()
    }

    fn advance_state(&self, state: &mut Box<dyn State<A>>, actions: &A) {
        state.mutate_on(actions);
    }
}

impl<A, C> Aggregator<A> for EndsOnContestWin<A, C>
where A: ActionType + Clone + 'static,
      C: Discounter + StateIterator<A> + Clone + 'static
{
    fn n(&self) -> usize {
        self.child.n()
    }
    fn u_i(&self, i: usize, strategies: &Strategies<A>) -> f64 {
        let actions_seq = strategies.actions();
        let state = &mut self.state0().clone();
        let gamma = self.child.gammas()[i];
        let mut proba = 1.;  // probability that nobody has won yet
        let mut u = 0.;
        for (t, actions) in actions_seq.iter().enumerate() {
            let payoff_func = state.belief(i).downcast_ref::<ModularPayoff<A>>().expect(
                "Belief should be ModularPayoff, but found something else"
            );
            u += proba * gamma.powi(t.try_into().unwrap()) * payoff_func.u_i(i, actions);
            if t != strategies.t() - 1 {
                // update proba
                let (_, p) = payoff_func.prod_func.f(actions);
                proba *= 1. - payoff_func.csf.q(p.view()).iter().sum::<f64>();
                // update state
                self.advance_state(state, actions);
            }
        }
        u
    }
    fn u(&self, strategies: &Strategies<A>) -> Array<f64, Ix1> {
        let actions_seq = strategies.actions();
        let state = &mut self.state0().clone();
        let gammas = self.child.gammas();
        let mut probas = vec![1.; gammas.len()];
        let mut u: Array<f64, Ix1> = Array::zeros(gammas.len());
        for (t, actions) in actions_seq.iter().enumerate() {
            u.iter_mut().zip(gammas.iter()).enumerate().for_each(|(i, (u_i, gamma))| {
                let payoff_func = state.belief(i).downcast_ref::<ModularPayoff<A>>().expect(
                    "Belief should be ModularPayoff, but found something else"
                );
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


#[derive(Clone)]
pub struct SolverResult<A: ActionType> {
    pub status: String,
    pub strategies: Option<Strategies<A>>,
}
