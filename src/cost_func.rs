use dyn_clone::DynClone;
use numpy::ndarray::{Array, Ix1};

use crate::strategies::*;

pub trait CostFunc<A: ActionType>: DynClone + Send + Sync {

    fn c_i(&self, i: usize, actions: &A) -> f64;
    fn c(&self, actions: &A) -> Array<f64, Ix1> {
        Array::from_iter((0..actions.n()).map(|i| self.c_i(i, actions)))
    }

    fn n(&self) -> usize;
}

impl<A: ActionType> CostFunc<A> for Box<dyn CostFunc<A>> {
    fn c_i(&self, i: usize, actions: &A) -> f64 { self.as_ref().c_i(i, actions) }
    fn c(&self, actions: &A) -> Array<f64, Ix1> { self.as_ref().c(actions) }
    fn n(&self) -> usize { self.as_ref().n() }
}

// long form of clone_trait_object!(CostFunc<A>):
impl<A: ActionType> Clone for Box<dyn CostFunc<A>> {
    fn clone(&self) -> Self {
        dyn_clone::clone_box(&**self)
    }
}

#[derive(Clone, Debug)]
pub struct FixedUnitCost {
    pub r: Array<f64, Ix1>,
}

impl FixedUnitCost {
    pub fn from_elem(n: usize, r: f64) -> FixedUnitCost {
        FixedUnitCost {
            r: Array::from_elem(n, r)
        }
    }
}

impl CostFunc<Actions> for FixedUnitCost {

    fn c_i(&self, i: usize, actions: &Actions) -> f64 {
        self.r[i] * (actions.xs()[i] + actions.xp()[i])
    }
    
    fn n(&self) -> usize {
        self.r.len()
    }
}

#[derive(Clone, Debug)]
pub struct FixedInvestCost {
    n: usize,
    pub r_x: Array<f64, Ix1>,
    pub r_inv: Array<f64, Ix1>,
}

impl FixedInvestCost {
    pub fn new(r_x: Array<f64, Ix1>, r_inv: Array<f64, Ix1>) -> FixedInvestCost {
        assert_eq!(r_x.len(), r_inv.len());
        FixedInvestCost {
            n: r_x.len(),
            r_x,
            r_inv,
        }
    }
    pub fn from_elems(n: usize, r_x: f64, r_inv: f64) -> FixedInvestCost {
        FixedInvestCost {
            n,
            r_x: Array::from_elem(n, r_x),
            r_inv: Array::from_elem(n, r_inv),
        }
    }
}

impl CostFunc<Actions> for FixedInvestCost {

    fn c_i(&self, i: usize, actions: &Actions) -> f64 {
        self.r_x[i] * (actions.xs()[i] + actions.xp()[i])
    }
    
    fn n(&self) -> usize {
        self.n
    }
}

impl CostFunc<InvestActions> for FixedInvestCost {

    fn c_i(&self, i: usize, actions: &InvestActions) -> f64 {
        self.r_x[i] * (actions.xs()[i] + actions.xp()[i]) + self.r_inv[i] * (actions.inv_s()[i] + actions.inv_p()[i])
    }

    fn n(&self) -> usize {
        self.n
    }
}
