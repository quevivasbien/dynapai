use std::fmt;
use numpy::ndarray::{Array, ArrayView, Ix1, Ix2, Ix3, s};
use dyn_clone::{DynClone, clone_trait_object};
use ndarray_rand::{RandomExt, rand_distr::LogNormal};


pub trait ActionType: DynClone + Send + Sync {
    fn data(&self) -> &Array<f64, Ix2>;
    fn data_mut(&mut self) -> &mut Array<f64, Ix2>;
    fn n(&self) -> usize { self.data().shape()[0] }

    fn from_array(data: Array<f64, Ix2>) -> Self where Self: Sized;
    fn nparams() -> usize where Self: Sized;

    fn xs(&self) -> ArrayView<f64, Ix1> { self.data().slice(s![.., 0]) }
    fn xp(&self) -> ArrayView<f64, Ix1> { self.data().slice(s![.., 1]) }
}

clone_trait_object!(ActionType);

impl fmt::Display for dyn ActionType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Actions: xs = {}, xp = {}", self.xs(), self.xp())
    }
}

pub trait InvestActionType: ActionType {
    fn inv_s(&self) -> ArrayView<f64, Ix1> { self.data().slice(s![.., 2]) }
    fn inv_p(&self) -> ArrayView<f64, Ix1> { self.data().slice(s![.., 3]) }
}

clone_trait_object!(InvestActionType);

impl fmt::Display for dyn InvestActionType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "InvestActions: xs = {}, xp = {}, inv_s = {}, inv_p = {}", self.xs(), self.xp(), self.inv_s(), self.inv_p())
    }
}

#[derive(Clone)]
pub struct Actions(Array<f64, Ix2>);

impl ActionType for Actions {
    fn data(&self) -> &Array<f64, Ix2> { &self.0 }
    fn data_mut(&mut self) -> &mut Array<f64, Ix2> { &mut self.0 }

    fn from_array(data: Array<f64, Ix2>) -> Self {
        assert_eq!(data.shape()[1], 2);
        Self(data)
    }
    fn nparams() -> usize { 2 }
}

#[derive(Clone)]
pub struct InvestActions(Array<f64, Ix2>);

impl ActionType for InvestActions {
    fn data(&self) -> &Array<f64, Ix2> { &self.0 }
    fn data_mut(&mut self) -> &mut Array<f64, Ix2> { &mut self.0 }

    fn from_array(data: Array<f64, Ix2>) -> Self {
        assert_eq!(data.shape()[1], 4);
        Self(data)
    }
    fn nparams() -> usize { 4 }
}

impl InvestActionType for InvestActions {}

#[derive(Clone)]
pub struct Strategies<A: ActionType>(Vec<A>);

impl<A: ActionType> Strategies<A> {

    pub fn t(&self) -> usize { self.actions().len() }
    pub fn n(&self) -> usize { self.actions()[0].n() }
    
    pub fn actions(&self) -> &Vec<A> { &self.0 }
    pub fn into_actions(self) -> Vec<A> { self.0 }
    pub fn actions_mut(&mut self) -> &mut Vec<A> { &mut self.0 }
    pub fn data(&self) -> Array<f64, Ix3> {
        Array::from_shape_vec(
            (self.t(), self.n(), A::nparams()),
            self.0.iter().flat_map(|a| a.data().iter().cloned()).collect()
        ).unwrap()
    }
    // set player i's strategy
    // given x is a 2d array of shape (t, nparams)
    pub fn set_i(&mut self, i: usize, x: Array<f64, Ix2>) {
        self.0.iter_mut().enumerate().for_each(|(j, a)|
            a.data_mut().slice_mut(s![i, ..]).assign(&x.slice(s![j, ..]))
        );
    }

    pub fn from_actions(actions: Vec<A>) -> Self {
        assert!(actions.iter().all(|a| a.n() == actions[0].n()));
        Self(actions)
    }
    pub fn random(t: usize, n: usize, mu: f64, sigma: f64) -> Result<Self, String> {
        let dist = match LogNormal::new(mu, sigma) {
            Ok(d) => d,
            Err(e) => return Err(format!("Error when creating LogNormal distribution: {}", e))
        };
        Ok(Self((0..t).map(|_|
            <A as ActionType>::from_array(Array::random((n, <A as ActionType>::nparams()), dist))
        ).collect()))
    }
}

pub trait MutatesOn<A> {
    fn mutate_on(&mut self, _actions: &A) {}
}
