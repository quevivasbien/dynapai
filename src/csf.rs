use dyn_clone::{DynClone, clone_trait_object};
use numpy::ndarray::{Array, ArrayView, Ix1};

pub trait CSF: DynClone + Send + Sync {
    fn q_i(&self, i: usize, p: ArrayView<f64, Ix1>) -> f64;
    fn q(&self, p: ArrayView<f64, Ix1>) -> Array<f64, Ix1> {
        Array::from_iter((0..p.len()).map(|i| self.q_i(i, p)))
    }
}

clone_trait_object!(CSF);

impl CSF for Box<dyn CSF> {
    fn q_i(&self, i: usize, p: ArrayView<f64, Ix1>) -> f64 {
        self.as_ref().q_i(i, p)
    }
    fn q(&self, p: ArrayView<f64, Ix1>) -> Array<f64, Ix1> {
        self.as_ref().q(p)
    }
}

#[derive(Clone)]
pub struct DefaultCSF;

impl CSF for DefaultCSF {
    fn q_i(&self, i: usize, p: ArrayView<f64, Ix1>) -> f64 {
        let sum_p: f64 = p.iter().sum();
        if sum_p == 0.0 {
            0.0
        } else {
            p[i] / sum_p
        }
    }

    fn q(&self, p: ArrayView<f64, Ix1>) -> Array<f64, Ix1> {
        let sum_p: f64 = p.iter().sum();
        if sum_p == 0.0 {
            Array::zeros(p.len())
        } else {
            Array::from_iter(p.iter().map(|x| x / sum_p))
        }
    }
}


#[derive(Clone, Debug)]
pub struct MaybeNoWinCSF { scale: f64 }

impl MaybeNoWinCSF {
    pub fn new(scale: f64) -> Self {
        Self { scale }
    }
    pub fn default() -> Self {
        Self { scale: 1.0 }
    }
}

impl CSF for MaybeNoWinCSF {
    fn q_i(&self, i: usize, p: ArrayView<f64, Ix1>) -> f64 {
        self.scale * p[i] / (1. + self.scale * p.iter().sum::<f64>())
    }
    fn q(&self, p: ArrayView<f64, Ix1>) -> Array<f64, Ix1> {
        let sum_p = p.iter().sum::<f64>();
        Array::from_iter(p.iter().map(|x|
            self.scale * x / (1. + self.scale * sum_p)
        ))
    }
}
