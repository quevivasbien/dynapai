use core::fmt;

use numpy::ndarray::{Array, ArrayView, Ix1, Ix2, s};
use dyn_clone::{DynClone, clone_trait_object};

pub trait ActionType: DynClone + Send + Sync {
    fn data(&self) -> &Array<f64, Ix2>;
    fn data_mut(&mut self) -> &mut Array<f64, Ix2>;
    fn n(&self) -> usize { self.data().shape()[0] }

    fn xs(&self) -> ArrayView<f64, Ix1> { self.data().slice(s![.., 0]) }
    fn xp(&self) -> ArrayView<f64, Ix1> { self.data().slice(s![.., 1]) }
}

impl ActionType for Box<dyn ActionType> {
    fn data(&self) -> &Array<f64, Ix2> { self.as_ref().data() }
    fn data_mut(&mut self) -> &mut Array<f64, Ix2> { self.as_mut().data_mut() }
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

impl ActionType for Box<dyn InvestActionType> {
    fn data(&self) -> &Array<f64, Ix2> { self.as_ref().data() }
    fn data_mut(&mut self) -> &mut Array<f64, Ix2> { self.as_mut().data_mut() }
}

impl InvestActionType for Box<dyn InvestActionType> {
    fn inv_s(&self) -> ArrayView<f64, Ix1> { self.as_ref().inv_s() }
    fn inv_p(&self) -> ArrayView<f64, Ix1> { self.as_ref().inv_p() }
}

clone_trait_object!(InvestActionType);

impl fmt::Display for dyn InvestActionType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "InvestActions: xs = {}, xp = {}, inv_s = {}, inv_p = {}", self.xs(), self.xp(), self.inv_s(), self.inv_p())
    }
}

#[derive(Clone)]
pub struct Actions(Array<f64, Ix2>);

impl Actions {
    pub fn from_array(data: Array<f64, Ix2>) -> Self {
        assert_eq!(data.shape()[1], 2);
        Self(data)
    }
}

impl ActionType for Actions {
    fn data(&self) -> &Array<f64, Ix2> { &self.0 }
    fn data_mut(&mut self) -> &mut Array<f64, Ix2> { &mut self.0 }
}

#[derive(Clone)]
pub struct InvestActions(Array<f64, Ix2>);

impl InvestActions {
    pub fn from_array(data: Array<f64, Ix2>) -> Self {
        assert_eq!(data.shape()[1], 4);
        Self(data)
    }
}

impl ActionType for InvestActions {
    fn data(&self) -> &Array<f64, Ix2> { &self.0 }
    fn data_mut(&mut self) -> &mut Array<f64, Ix2> { &mut self.0 }
}

impl InvestActionType for InvestActions {}

#[derive(Clone)]
pub struct Strategies<A: ActionType>(Vec<A>);

impl<A: ActionType> Strategies<A> {

    pub fn actions(&self) -> &Vec<A> { &self.0 }
    pub fn t(&self) -> usize { self.actions().len() }
    pub fn n(&self) -> usize { self.actions()[0].n() }
    pub fn from_actions(actions: Vec<A>) -> Self {
        assert!(actions.iter().all(|a| a.n() == actions[0].n()));
        Self(actions)
    }
}

pub trait MutatesOn<A> {
    fn mutate_on(&mut self, _actions: &A) {}
}
