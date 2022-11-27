use numpy::ndarray::{Array, Ix1};
use dyn_clone::DynClone;
use std::fmt;

use crate::strategies::*;

pub trait ProdFunc<A: ActionType>: DynClone + MutatesOn<A> + Send + Sync {
    fn f_i(&self, i: usize, actions: &A) -> (f64, f64);
    fn f(&self, actions: &A) -> (Array<f64, Ix1>, Array<f64, Ix1>) {
        let (s, p) = (0..actions.n()).map(|i| self.f_i(i, actions)).unzip();
        (Array::from_vec(s), Array::from_vec(p))
    }

    fn n(&self) -> usize;
}

// long form of clone_trait_object!(ProdFunc<A>):
impl<A: ActionType> Clone for Box<dyn ProdFunc<A>> {
    fn clone(&self) -> Self {
        dyn_clone::clone_box(&**self)
    }
}


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
impl ProdFunc<Actions> for DefaultProd {
    fn f_i(&self, i: usize, actions: &Actions) -> (f64, f64) {
        self._f_i(i, actions)
    }
    fn n(&self) -> usize {
        self.n
    }
}

impl<A: InvestActionType + Clone> ProdFunc<A> for DefaultProd {
    fn f_i(&self, i: usize, actions: &A) -> (f64, f64) {
        self._f_i(i, actions)
    }
    fn n(&self) -> usize {
        self.n
    }
}

impl MutatesOn<Actions> for DefaultProd {}

impl<A: InvestActionType> MutatesOn<A> for DefaultProd {
    fn mutate_on(&mut self, actions: &A) {
        self.a.iter_mut().zip(actions.inv_s().iter()).for_each(
            |(a, inv_s)| *a += inv_s
        );
        self.b.iter_mut().zip(actions.inv_p().iter()).for_each(
            |(b, inv_p)| *b += inv_p
        );
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
