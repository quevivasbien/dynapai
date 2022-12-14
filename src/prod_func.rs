use downcast_rs::{Downcast, impl_downcast};
use dyn_clone::{DynClone, clone_trait_object};
use numpy::ndarray::{Array, Ix1};
use std::fmt;

use crate::prelude::*;


pub trait ProdFunc<A: ActionType>: DynClone + Downcast + MutatesOn<A> + Send + Sync {
    fn f_i(&self, i: usize, actions: &A) -> (f64, f64);
    fn f(&self, actions: &A) -> (Array<f64, Ix1>, Array<f64, Ix1>) {
        let (s, p) = (0..actions.n()).map(|i| self.f_i(i, actions)).unzip();
        (Array::from_vec(s), Array::from_vec(p))
    }

    fn n(&self) -> usize;
}

clone_trait_object!(<A> ProdFunc<A> where A: ActionType);
impl_downcast!(ProdFunc<A> where A: ActionType);


#[derive(Clone, Debug)]
pub struct DefaultProd {
    n: usize,
    pub a: Array<f64, Ix1>,
    pub alpha: Array<f64, Ix1>,
    pub b: Array<f64, Ix1>,
    pub beta: Array<f64, Ix1>,
}

impl DefaultProd {
    pub fn new(a: Array<f64, Ix1>, alpha: Array<f64, Ix1>, b: Array<f64, Ix1>, beta: Array<f64, Ix1>) -> Result<DefaultProd, &'static str> {
        let n = a.len();
        if n != alpha.len() || n != b.len() || n != beta.len() {
            return Err("When creating new DefaultProd: All input arrays must have the same length");
        }
        Ok(DefaultProd { n, a, alpha, b, beta })
    }

    fn _f_i(&self, i: usize, actions: &dyn ActionType) -> (f64, f64) {
        (
            self.a[i] * actions.xs()[i].powf(self.alpha[i]),
            self.b[i] * actions.xp()[i].powf(self.beta[i])
        )
    }
}

// need to do this silliness since MutatesOn<A> is not defined for all ProdFunc types
macro_rules! default_prod_impl {
    ($($a:ty),*) => {
        $(impl ProdFunc<$a> for DefaultProd {
            fn f_i(&self, i: usize, actions: &$a) -> (f64, f64) {
                self._f_i(i, actions)
            }

            fn n(&self) -> usize { self.n }
        })*
    };
}

default_prod_impl!(Actions, InvestActions, SharingActions);

impl MutatesOn<Actions> for DefaultProd {}

impl MutatesOn<InvestActions> for DefaultProd {
    fn mutate_on(&mut self, actions: &InvestActions) {
        self.a.iter_mut().zip(actions.inv_s().iter()).for_each(
            |(a, inv_s)| *a += inv_s
        );
        self.b.iter_mut().zip(actions.inv_p().iter()).for_each(
            |(b, inv_p)| *b += inv_p
        );
    }
}

impl MutatesOn<SharingActions> for DefaultProd {
    fn mutate_on(&mut self, actions: &SharingActions) {
        let old_a = self.a.clone();
        let inv_s = actions.inv_s();
        let share_s = actions.share_s();
        let old_b = self.b.clone();
        let inv_p = actions.inv_p();
        let share_p = actions.share_p();
        self.a.iter_mut().zip(self.b.iter_mut()).enumerate().for_each(|(i, (a, b))| {
            *a += inv_s[i] + old_a.iter().zip(share_s.iter()).map(|(a_, sh_s)|
                f64::max(0., positive_bound(*sh_s) * (a_ - old_a[i]))
            ).sum::<f64>();
            *b += inv_p[i] + old_b.iter().zip(share_p.iter()).map(|(b_, sh_p)|
                f64::max(0., positive_bound(*sh_p) * (b_ - old_b[i]))
            ).sum::<f64>();
        });
    }
}

impl fmt::Display for DefaultProd {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f, "DefaultProd {{ a = {}, alpha = {}, b = {}, beta = {} }}",
            self.a, self.alpha, self.b, self.beta
        )
    }
}
