use downcast_rs::{Downcast, impl_downcast};
use dyn_clone::{clone_trait_object, DynClone};
use numpy::ndarray::{Array, ArrayView, Ix1};

pub trait DisasterCost: DynClone + Downcast + Send + Sync {
    fn d_i(&self, i: usize, s: ArrayView<f64, Ix1>, p: ArrayView<f64, Ix1>) -> f64;
    fn d(&self, s: ArrayView<f64, Ix1>, p: ArrayView<f64, Ix1>) -> Array<f64, Ix1> {
        Array::from_iter((0..s.len()).map(|i| self.d_i(i, s, p)))
    }

    fn n(&self) -> usize;
}

clone_trait_object!(DisasterCost);
impl_downcast!(DisasterCost);


#[derive(Clone)]
pub struct ConstantDisasterCost {
    pub d: Array<f64, Ix1>,
}

impl DisasterCost for ConstantDisasterCost {
    fn d_i(&self, i: usize, _s: ArrayView<f64, Ix1>, _p: ArrayView<f64, Ix1>) -> f64 {
        self.d[i]
    }

    fn n(&self) -> usize {
        self.d.len()
    }
}

impl ConstantDisasterCost {
    pub fn new(n: usize, d: f64) -> Self {
        ConstantDisasterCost {
            d: Array::from_elem(n, d),
        }
    }
}
