pub use pyo3::prelude::*;
pub use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};
pub use numpy::ndarray::{stack, Axis};

pub use crate::prelude::*;

pub mod aggregator;
pub mod payoff_func;
pub mod prod_func;
pub mod strategies;

pub use aggregator::*;
pub use payoff_func::*;
pub use prod_func::*;
pub use strategies::*;

#[macro_export]
macro_rules! unpack_py {
    { $input:ident => $output:ident [ $($typ:ty)|+ ]; $exec:expr } => {
        $(if let Ok($output) = $input.extract::<$typ>() {
            $exec
        }) else +
        else {
            panic!("Invalid input type; expected one of: {}",
                std::any::type_name::<($($typ),+)>()
            );
        }
    }
}