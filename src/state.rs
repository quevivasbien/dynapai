use downcast_rs::{Downcast, impl_downcast};
use dyn_clone::{DynClone, clone_trait_object};

use crate::prelude::*;


pub trait State<A: ActionType>: DynClone + Downcast + MutatesOn<A> + Send + Sync {
    fn n(&self) -> usize;
    fn belief(&self, i: usize) -> &Box<dyn PayoffFunc<A>>;
}

clone_trait_object!(<A> State<A> where A: ActionType);
impl_downcast!(State<A> where A: ActionType);

#[derive(Clone)]
pub struct CommonBeliefs<A: ActionType>(pub Box<dyn PayoffFunc<A>>);

impl<A: ActionType> MutatesOn<A> for CommonBeliefs<A> {
    fn mutate_on(&mut self, actions: &A) {
        self.0.mutate_on(actions)
    }
}

impl<A: ActionType + Clone + 'static> State<A> for CommonBeliefs<A> {
    fn n(&self) -> usize {
        self.0.n()
    }
    fn belief(&self, _i: usize) -> &Box<dyn PayoffFunc<A>> {
        &self.0
    }
} 


#[derive(Clone)]
pub struct HetBeliefs<A: ActionType> {
    n: usize,
    beliefs: Vec<Box<dyn PayoffFunc<A>>>,
}

impl<A: ActionType + 'static> HetBeliefs<A> {
    pub fn new(beliefs: Vec<Box<dyn PayoffFunc<A>>>) -> Result<HetBeliefs<A>, &'static str> {
        if beliefs.len() == 0 {
            return Err("When creating new HetBeliefs: beliefs must have length > 0");
        }
        if beliefs.iter().any(|b| b.n() != beliefs.len()) {
            return Err("When creating new HetBeliefs: All beliefs must have the same n, matching length of beliefs");
        }
        Ok(HetBeliefs { n: beliefs.len(), beliefs })
    }
}

impl<A: ActionType> MutatesOn<A> for HetBeliefs<A> {
    fn mutate_on(&mut self, actions: &A) {
        for i in 1..self.n {
            self.beliefs[i].mutate_on(actions);
        }
    }
}

impl<A: ActionType + Clone + 'static> State<A> for HetBeliefs<A> {
    fn n(&self) -> usize {
        self.n
    }
    fn belief(&self, i: usize) -> &Box<dyn PayoffFunc<A>> {
        &self.beliefs[i]
    }
}

pub trait StateIterator<A: ActionType>: DynClone + Send + Sync
{
    fn state0(&self) -> &Box<dyn State<A>>;
    fn advance_state(&self, _state: &mut Box<dyn State<A>>, _actions: &A) {}
}

impl<A: ActionType> Clone for Box<dyn StateIterator<A>>
{
    fn clone(&self) -> Self {
        dyn_clone::clone_box(&**self)
    }
}
