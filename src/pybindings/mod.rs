pub use pyo3::prelude::*;
pub use pyo3::types::PyList;
pub use numpy::{PyArray1, PyArray2, PyReadonlyArray2, IntoPyArray};
pub use numpy::ndarray::{stack, Array, Array1, Axis};

pub use crate::prelude::*;

pub mod aggregator;
pub mod cost_func;
pub mod csf;
pub mod payoff_func;
pub mod prod_func;
pub mod reward_func;
pub mod risk_func;
pub mod state;
pub mod strategies;

pub use aggregator::*;
pub use cost_func::*;
pub use csf::*;
pub use payoff_func::*;
pub use prod_func::*;
pub use reward_func::*;
pub use risk_func::*;
pub use state::*;
pub use strategies::*;

// this macro takes an input ($input) of type &PyAny
// and attempts to downcast it to a variable of type $typ
// before plugging it into the expression $exec.
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
    // a special variant when we want to package the output in an enum
    { $input:expr => $output:ident [ $basic_typ:ty | $invest_typ:ty ]; $exec:expr => $enumout:ident } => {
        if let Ok($output) = $input.extract::<$basic_typ>() {
            $enumout::Basic($exec)
        }
        else if let Ok($output) = $input.extract::<$invest_typ>() {
            $enumout::Invest($exec)
        }
        else {
            panic!("Invalid input type; expected one of: {}",
                std::any::type_name::<($basic_typ, $invest_typ)>()
            );
        }
    };
}

// This macro takes a container enum with Basic & Invest variants,
// unpacks them into a variable $name,
// and executes the given expression using that $name.
// The idea of this and the following macros is to be able to implement
// new action types and only have to change the macro definitions
// instead of a bunch of code in the pybindings modules.
#[macro_export]
macro_rules! unpack_py_enum {
    ( $in:expr => $enumname:ident($name:ident); $exec:expr ) => {
        match $in {
            $enumname::Basic($name) => $exec,
            $enumname::Invest($name) => $exec
        }
    };
    ( $in:expr => $enumname:ident($name:ident); $exec:expr => $outenumname:ident ) => {
        match $in {
            $enumname::Basic($name) => $outenumname::Basic($exec),
            $enumname::Invest($name) => $outenumname::Invest($exec)
        }
    };
    (
        $in:expr => $enumname:ident($name:ident);
        $pytype_in:expr => $pytype_out:ident [ $BasicType:ty | $InvestType:ty ];
        $exec:expr
    ) => {
        match $in {
            $enumname::Basic($name) => {
                $crate::unpack_py! { $pytype_in => $pytype_out [ $BasicType ]; $exec }
            },
            $enumname::Invest($name) => {
                $crate::unpack_py! { $pytype_in => $pytype_out [ $InvestType ]; $exec }
            }
        }
    };
    (
        $in:expr => $enumname:ident($name:ident);
        $pytype_in:expr => $pytype_out:ident [ $BasicType:ty | $InvestType:ty ];
        $exec:expr => $outenumname:ident
    ) => {
        match $in {
            $enumname::Basic($name) => {
                $crate::unpack_py! { $pytype_in => $pytype_out [ $BasicType ]; $outenumname::Basic($exec) }
            },
            $enumname::Invest($name) => {
                $crate::unpack_py! { $pytype_in => $pytype_out [ $InvestType ]; $outenumname::Invest($exec) }
            }
        }
    };
}

#[macro_export]
macro_rules! unpack_py_enum_on_actions {
    (
        $in:expr => $enumname:ident($name:ident);
        $actions_in:expr => $actions_out:ident;
        $exec:expr
    ) => {
        $crate::unpack_py_enum! {
            $in => $enumname($name);
            $actions_in => $actions_out [PyActions | PyInvestActions];
            {
                let $actions_out = $actions_out.unpack();
                $exec
            }
        }
    }
}

#[macro_export]
macro_rules! unpack_py_enum_on_strategies {
    (
        $in:expr => $enumname:ident($name:ident);
        $strategies_in:expr => $strategies_out:ident;
        $exec:expr
    ) => {
        $crate::unpack_py_enum! {
            $in => $enumname($name);
            $strategies_in => $strategies_out [Vec<PyActions> | Vec<PyInvestActions>];
            {
                let $strategies_out = Strategies::from_actions(
                        $strategies_out.into_iter().map(|pyactions|
                            pyactions.unpack()
                        ).collect()
                    );
                $exec
            }
        }
    }
}

#[macro_export]
macro_rules! def_py_enum {
    ($name:ident($contenttype:ident)) => {
        #[derive(Clone)]
        pub enum $name {
            Basic($contenttype<Actions>),
            Invest($contenttype<InvestActions>)
        }
    };
    ($name:ident(Box<dyn $contenttype:ident>)) => {
        #[derive(Clone)]
        pub enum $name {
            Basic(Box<dyn $contenttype<Actions>>),
            Invest(Box<dyn $contenttype<InvestActions>>)
        }
    };
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
