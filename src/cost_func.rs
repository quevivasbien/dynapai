use downcast_rs::{Downcast, impl_downcast};
use dyn_clone::{DynClone, clone_trait_object};
use numpy::{ndarray::{Array, Ix1, s}, Ix2};

use crate::strategies::*;

pub trait CostFunc<A: ActionType>: DynClone + Downcast + Send + Sync {

    fn c_i(&self, i: usize, actions: &A) -> f64;
    fn c(&self, actions: &A) -> Array<f64, Ix1> {
        Array::from_iter((0..actions.n()).map(|i| self.c_i(i, actions)))
    }

    fn n(&self) -> usize;
}

clone_trait_object!(<A> CostFunc<A> where A: ActionType);
impl_downcast!(CostFunc<A> where A: ActionType);

pub trait FixedCost<A: ActionType>: CostFunc<A> {
    fn nparams() -> usize where Self: Sized;
    fn r(&self) -> &Array<f64, Ix2>;
    fn r_mut(&mut self) -> &mut Array<f64, Ix2>;
    fn new_unchecked(r: Array<f64, Ix2>) -> Self where Self: Sized;
    
    fn new(r: Array<f64, Ix2>) -> Result<Self, String> where Self: Sized {
        if r.shape()[1] != Self::nparams() {
            Err(format!(
                "Invalid number of params: {}, expected {}",
                r.shape()[1], Self::nparams()
            ))
        } else {
            Ok(Self::new_unchecked(r))
        }
    }
    fn from_params(n: usize, params: Vec<f64>) -> Self where Self: Sized {
        let r = Array::from_shape_fn(
            (n, params.len()),
            |(_i, j)| params[j]
        );
        Self::new_unchecked(r)
    }
}

impl<A: ActionType, C: FixedCost<A>> CostFunc<A> for C {
    fn c_i(&self, i: usize, actions: &A) -> f64 {
        self.r().slice(s![i, ..]).dot(&actions.data().slice(s![i, ..]))
    }

    fn n(&self) -> usize {
        self.r().shape()[0]
    }
}

clone_trait_object!(<A> FixedCost<A> where A: ActionType);
impl_downcast!(FixedCost<A> where A: ActionType);

macro_rules! impl_fixed_cost {
    ($name:ident, $a_type:ident) => {
        #[derive(Clone)]
        pub struct $name(pub Array<f64, Ix2>);

        impl FixedCost<$a_type> for $name {
            fn nparams() -> usize {
                $a_type::nparams()
            }
            fn r(&self) -> &Array<f64, Ix2> {
                &self.0
            }
            fn r_mut(&mut self) -> &mut Array<f64, Ix2> {
                &mut self.0
            }
            fn new_unchecked(r: Array<f64, Ix2>) -> Self {
                Self(r)
            }
        }
    };
}

impl_fixed_cost!(BasicFixedCost, Actions);
impl_fixed_cost!(InvestFixedCost, InvestActions);
impl_fixed_cost!(SharingFixedCost, SharingActions);
