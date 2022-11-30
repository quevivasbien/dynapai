use downcast_rs::{Downcast, impl_downcast};
use dyn_clone::{DynClone, clone_trait_object};
use numpy::ndarray::{Array, ArrayView, Ix1};
use std::fmt;

pub trait RewardFunc: DynClone + Downcast + Send + Sync {
    fn win_i(&self, i: usize, p: ArrayView<f64, Ix1>) -> f64;
    fn lose_i(&self, i: usize, p: ArrayView<f64, Ix1>) -> f64;
    fn reward(&self, i: usize, p: ArrayView<f64, Ix1>) -> Array<f64, Ix1> {
        Array::from_iter((0..p.len()).map(|j| {
            if j == i {
                self.win_i(j, p)
            } else {
                self.lose_i(j, p)
            }
        }))
    }

    fn n(&self) -> usize;
}

clone_trait_object!(RewardFunc);
impl_downcast!(RewardFunc);


#[derive(Clone)]
pub struct LinearReward {
    n: usize,
    pub win_a: Array<f64, Ix1>,
    pub win_b: Array<f64, Ix1>,
    pub lose_a: Array<f64, Ix1>,
    pub lose_b: Array<f64, Ix1>,
}

impl LinearReward {
    pub fn new(
        win_a: Array<f64, Ix1>,
        win_b: Array<f64, Ix1>,
        lose_a: Array<f64, Ix1>,
        lose_b: Array<f64, Ix1>
    ) -> Result<Self, &'static str> {
        let n = win_a.len();
        if win_b.len() != n || lose_a.len() != n || lose_b.len() != n {
            return Err("When creating LinearReward: All input arrays must have the same length");
        }
        Ok(LinearReward { n, win_a, win_b, lose_a, lose_b, })
    }

    pub fn default(n: usize) -> Self {
        LinearReward::new(
            Array::ones(n),
            Array::zeros(n),
            Array::zeros(n),
            Array::zeros(n),
        ).unwrap()
    }
}

impl RewardFunc for LinearReward {
    fn win_i(&self, i: usize, p: ArrayView<f64, Ix1>) -> f64 {
        self.win_a[i] + self.win_b[i] * p[i]
    }
    fn lose_i(&self, i: usize, p: ArrayView<f64, Ix1>) -> f64 {
        self.lose_a[i] + self.lose_b[i] * p[i]
    }

    fn n(&self) -> usize {
        self.n
    }
}

impl fmt::Display for LinearReward {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f, "LinearReward {{ win_a: {}, win_b: {}, lose_a: {}, lose_b: {} }}",
            self.win_a, self.win_b, self.lose_a, self.lose_b
        )
    }
}
