use std::fmt;

use downcast_rs::{Downcast, impl_downcast};
use dyn_clone::{DynClone, clone_trait_object};
use numpy::ndarray::{ArrayView, Ix1, Array};

pub trait RiskFunc: DynClone + Downcast + Send + Sync {
    // sigma_i is proba(safe | i wins)
    fn sigma_i(&self, i: usize, s: ArrayView<f64, Ix1>, p: ArrayView<f64, Ix1>) -> f64;
    fn sigma(&self, s: ArrayView<f64, Ix1>, p: ArrayView<f64, Ix1>) -> Array<f64, Ix1> {
        Array::from_iter((0..s.len()).map(|i| self.sigma_i(i, s, p)))
    }

    fn n(&self) -> usize;
}

clone_trait_object!(RiskFunc);
impl_downcast!(RiskFunc);


pub trait RiskFuncWithTheta: RiskFunc {
    fn theta(&self) -> &Array<f64, Ix1>;
}

clone_trait_object!(RiskFuncWithTheta);


#[derive(Clone)]
pub struct WinnerOnlyRisk {
    pub theta: Array<f64, Ix1>,
}

impl WinnerOnlyRisk {
    pub fn new(n: usize, theta: f64) -> Result<Self, &'static str> {
        Ok(WinnerOnlyRisk {
            theta: Array::from_elem(n, theta),
        })
    }
}

impl RiskFunc for WinnerOnlyRisk {
    fn sigma_i(&self, i: usize, s: ArrayView<f64, Ix1>, p: ArrayView<f64, Ix1>) -> f64 {
        let s_ = s[i] * p[i].powf(-self.theta[i]);
        s_ / (1.0 + s_)
    }

    fn n(&self) -> usize {
        self.theta.len()
    }
}

impl RiskFuncWithTheta for WinnerOnlyRisk {
    fn theta(&self) -> &Array<f64, Ix1> {
        &self.theta
    }
}

impl fmt::Display for WinnerOnlyRisk {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "WinnerOnlyRisk: theta = {}", self.theta)
    }
}
