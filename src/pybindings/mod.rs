pub use pyo3::prelude::*;
pub use pyo3::types::PyList;
pub use numpy::{PyArray1, PyArray2, PyReadonlyArray2, IntoPyArray};
pub use numpy::ndarray::{stack, Array, Array1, Axis};

pub use crate::prelude::*;

pub mod aggregator;
pub mod csf;
pub mod payoff_func;
pub mod prod_func;
pub mod reward_func;
pub mod risk_func;
pub mod state;
pub mod strategies;

pub use aggregator::*;
pub use csf::*;
pub use payoff_func::*;
pub use prod_func::*;
pub use reward_func::*;
pub use risk_func::*;
pub use state::*;
pub use strategies::*;

#[macro_export]
macro_rules! unpack_py {
    // this second branch is only needed so linter doesn't complain about unused parens
    { $input:expr => $output:ident [ $typ:ty ]; $exec:expr } => {
        if let Ok($output) = $input.extract::<$typ>() {
            $exec
        }
        else {
            panic!("Invalid input type; expected: {}",
                std::any::type_name::<$typ>()
            );
        }
    };
    { $input:expr => $output:ident [ $($typ:ty)|+ ]; $exec:expr } => {
        $(if let Ok($output) = $input.extract::<$typ>() {
            $exec
        }) else +
        else {
            panic!("Invalid input type; expected one of: {}",
                std::any::type_name::<($($typ),+)>()
            );
        }
    };
}

#[macro_export]
macro_rules! unpack_py_on_actions {
    { $input:expr => $output:ident : $enumname:ident; $actin:expr => $actout:ident; $exec:expr } => {
        match $input {
            $enumname::Basic($output) => {
                crate::unpack_py! {
                    $actin => pyactions [PyActions];
                    {
                        let $actout = pyactions.unpack();
                        $exec
                    }
                }
            },
            $enumname::Invest($output) => {
                crate::unpack_py! {
                    $actin => pyactions [PyInvestActions];
                    {
                        let $actout = pyactions.unpack();
                        $exec
                    }
                }
            }
        }
    }
}

#[macro_export]
macro_rules! unpack_py_on_strategies {
    { $input:expr => $output:ident : $enumname:ident; $stratin:expr => $stratout:ident; $exec:expr } => {
        match $input {
            $enumname::Basic($output) => {
                crate::unpack_py! {
                    $stratin => actions_vec [Vec<PyActions>];
                    {
                        let $stratout = Strategies::from_actions(actions_vec.into_iter().map(|x| x.unpack()).collect());
                        $exec
                    }
                }
            },
            $enumname::Invest($output) => {
                crate::unpack_py! {
                    $stratin => actions_vec [Vec<PyInvestActions>];
                    {
                        let $stratout = Strategies::from_actions(actions_vec.into_iter().map(|x| x.unpack()).collect());
                        $exec
                    }
                }
            }
        }
    }
}

trait PyContainer {
    type Item;
    fn unpack(self) -> Self::Item;
    fn get(&self) -> &Self::Item;
}

// impl PyContainer for tuple struct of form Name(Contains)
#[macro_export]
macro_rules! pycontainer {
    { $name:ident($contains:ty) } => {
        impl PyContainer for $name {
            type Item = $contains;
            fn unpack(self) -> Self::Item {
                self.0
            }
            fn get(&self) -> &Self::Item {
                &self.0
            }
        }
    };
    { $name:ident($field:ident : $contains:ty) } => {
        impl PyContainer for $name {
            type Item = $contains;
            fn unpack(self) -> Self::Item {
                self.$field
            }
            fn get(&self) -> &Self::Item {
                &self.$field
            }
        }
    };
}

fn value_error<S>(msg: S) -> PyErr
where S: Into<String> + std::marker::Send + std::marker::Sync + pyo3::IntoPy<pyo3::Py<pyo3::PyAny>> + 'static
{
    pyo3::exceptions::PyValueError::new_err(msg).into()
}
