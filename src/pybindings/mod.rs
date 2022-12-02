pub use pyo3::prelude::*;
pub use pyo3::types::PyList;
use pyo3::exceptions::PyTypeError;
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


#[derive(PartialEq, Clone)]
enum ObjectType {
    Basic,
    Invest,
    Sharing,
}

impl std::fmt::Display for ObjectType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ObjectType::Basic => write!(f, "basic"),
            ObjectType::Invest => write!(f, "invest"),
            ObjectType::Sharing => write!(f, "sharing"),
        }
    }
}

trait HasObjectType {
    fn object_type(&self) -> ObjectType;
}

// defines an enum that holds variants of an object for each ObjectType variant
#[macro_export]
macro_rules! def_py_enum {
    ($name:ident($contenttype:ident)) => {
        #[derive(Clone)]
        pub enum $name {
            Basic($contenttype<Actions>),
            Invest($contenttype<InvestActions>),
            Sharing($contenttype<SharingActions>),
        }
        impl HasObjectType for $name {
            fn object_type(&self) -> ObjectType {
                match self {
                    $name::Basic(_) => ObjectType::Basic,
                    $name::Invest(_) => ObjectType::Invest,
                    $name::Sharing(_) => ObjectType::Sharing,
                }
            }
        }
    };
    ($name:ident(Box<dyn $contenttype:ident>)) => {
        #[derive(Clone)]
        pub enum $name {
            Basic(Box<dyn $contenttype<Actions>>),
            Invest(Box<dyn $contenttype<InvestActions>>),
            Sharing(Box<dyn $contenttype<SharingActions>>),
        }
        impl HasObjectType for $name {
            fn object_type(&self) -> ObjectType {
                match self {
                    $name::Basic(_) => ObjectType::Basic,
                    $name::Invest(_) => ObjectType::Invest,
                    $name::Sharing(_) => ObjectType::Sharing,
                }
            }
        }
    };
}


// This macro takes a container enum with Basic & Invest variants,
// unpacks them into a variable $name,
// and executes the given expression using that $name.
// The idea of this macro is to be able to implement
// new action types and only have to change this macro definition
// and the ObjectType-related code above
// instead of a bunch of code in the pybindings modules.
#[macro_export]
macro_rules! unpack_py_enum {
    // for when we want to execute the same code for all variants
    ( $in:expr => $enumname:ident($name:ident); $exec:expr ) => {
        match $in {
            $enumname::Basic($name) => $exec,
            $enumname::Invest($name) => $exec,
            $enumname::Sharing($name) => $exec,
        }
    };
    // for when we want to repackage result in new enum at the end
    ( $in:expr => $enumname:ident($name:ident); $exec:expr => $outenumname:ident ) => {
        match $in {
            $enumname::Basic($name) => $outenumname::Basic($exec),
            $enumname::Invest($name) => $outenumname::Invest($exec),
            $enumname::Sharing($name) => $outenumname::Sharing($exec),
        }
    };
    // for when we want to combine the enum with some other type
    (
        $in:expr => $enumname:ident($name:ident);
        $other_in:expr => $other_enumname:ident($other_name:ident);
        $exec:expr
    ) => {
        match $in {
            $enumname::Basic($name) => {
                match $other_in {
                    $other_enumname::Basic($other_name) => $exec,
                    _ => panic!("Invalid input type; expected Basic type"),
                }
            },
            $enumname::Invest($name) => {
                match $other_in {
                    $other_enumname::Invest($other_name) => $exec,
                    _ => panic!("Invalid input type; expected Invest type"),
                }
            },
            $enumname::Sharing($name) => {
                match $other_in {
                    $other_enumname::Sharing($other_name) => $exec,
                    _ => panic!("Invalid input type; expected Sharing type"),
                }
            },
        }
    };
    // for when we want to combine the enum with some other type
    // and repackage the result in an enum
    (
        $in:expr => $enumname:ident($name:ident);
        $other_in:expr => $other_enumname:ident($other_name:ident);
        $exec:expr => $outenumname:ident
    ) => {
        match $in {
            $enumname::Basic($name) => {
                match $other_in {
                    $other_enumname::Basic($other_name) => $outenumname($exec),
                    _ => panic!("Invalid input type; expected Basic type"),
                }
            },
            $enumname::Invest($name) => {
                match $other_in {
                    $other_enumname::Invest($other_name) => $outenumname($exec),
                    _ => panic!("Invalid input type; expected Invest type"),
                }
            },
            $enumname::Sharing($name) => {
                match $other_in {
                    $other_enumname::Sharing($other_name) => $outenumname($exec),
                    _ => panic!("Invalid input type; expected Sharing type"),
                }
            },
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
            $actions_in.get() => ActionContainer($actions_out);
            $exec
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
            $strategies_in.get() => StrategyContainer($strategies_out);
            $exec
        }
    }
}

#[macro_export]
macro_rules! unpack_py_enum_expect {
    ( $in:expr => $enumname:ident::$variant:ident($name:ident) ) => {
        match $in {
            $enumname::$variant($name) => Ok($name),
            _ => Err(PyErr::new::<PyTypeError, _>("Invalid input type")),
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
