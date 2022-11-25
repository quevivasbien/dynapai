use dyn_clone::DynClone;

use crate::prelude::*;


pub trait State: DynClone + Send + Sync {
    type PFunc: PayoffFunc;

    fn n(&self) -> usize;
    fn belief(&self, i: usize) -> &Self::PFunc;
}

impl<T: PayoffFunc> Clone for Box<dyn State<PFunc = T>> {
    fn clone(&self) -> Self {
        dyn_clone::clone_box(&**self)
    }
}

impl<T: PayoffFunc> State for Box<dyn State<PFunc = T>> {
    type PFunc: = T;
    fn n(&self) -> usize { self.as_ref().n() }
    fn belief(&self, i: usize) -> &T { self.as_ref().belief(i) }
}

// a PayoffFunc can itself act on a state,
// representing a state where everyone has the same beliefs
impl<T: PayoffFunc> State for T {
    type PFunc = T;

    fn n(&self) -> usize {
        self.n()
    }
    fn belief(&self, _i: usize) -> &Self {
        self
    }
}


#[derive(Clone)]
pub struct HetBeliefs<T: PayoffFunc> {
    n: usize,
    beliefs: Vec<T>,
}

impl<T: PayoffFunc> HetBeliefs<T> {
    pub fn new(beliefs: Vec<T>) -> Result<HetBeliefs<T>, &'static str> {
        if beliefs.len() == 0 {
            return Err("When creating new HetBeliefs: beliefs must have length > 0");
        }
        let n = beliefs[0].n();
        if beliefs.iter().any(|b| b.n() != n) {
            return Err("When creating new HetBeliefs: All beliefs must have the same n");
        }
        Ok(HetBeliefs { n, beliefs })
    }
}

impl<T: PayoffFunc + Clone> State for HetBeliefs<T> {
    type PFunc = T;

    fn n(&self) -> usize {
        self.n
    }
    fn belief(&self, i: usize) -> &T {
        &self.beliefs[i]
    }
}


pub trait StateIterator: DynClone + Send + Sync
{
    type St: State + Clone;
    fn state0(&self) -> &Self::St;
    fn advance_state(&self, _state: &mut Self::St, _actions: &<<Self::St as State>::PFunc as PayoffFunc>::Act) {}
}

impl<S> Clone for Box<dyn StateIterator<St = S>>
where S: State
{
    fn clone(&self) -> Self {
        dyn_clone::clone_box(&**self)
    }
}
