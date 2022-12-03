use crate::py::*;
use crate::{def_py_enum, pycontainer, unpack_py_enum};


def_py_enum!(StateContainer(Box<dyn State>));

#[derive(Clone)]
#[pyclass(name = "State")]
pub struct PyState {
    pub state: StateContainer,
    pub class: &'static str,
}
pycontainer!(PyState(state: StateContainer));


#[pymethods]
impl PyState {
    #[new]
    fn new(payoff_func: PyPayoffFunc) -> Self {
        Self::common_beliefs(payoff_func)
    }

    #[staticmethod]
    fn common_beliefs(payoff_func: PyPayoffFunc) -> Self {
        let state_container = unpack_py_enum! {
            [PayoffFuncContainer](payoff_func) = payoff_func.unpack();
            Box::new(CommonBeliefs(Box::new(payoff_func))) => StateContainer
        };
        Self {
            state: state_container,
            class: "CommonBeliefs",
        }
    }

    #[staticmethod]
    fn het_beliefs(beliefs: Vec<PyPayoffFunc>) -> Self {
        // check that all beliefs are the same type, and unpack them
        let state_container = if beliefs.iter().all(|b|
            match b.get() {
                PayoffFuncContainer::Basic(_) => true,
                _ => false,
            }
        ) {
            let beliefs = beliefs.into_iter().map(|b| match b.unpack() {
                PayoffFuncContainer::Basic(payoff_func) => Box::new(payoff_func) as Box<dyn PayoffFunc<Actions>>,
                _ => unreachable!(),
            }).collect();
            StateContainer::Basic(Box::new(HetBeliefs::new(beliefs).unwrap()))
        }
        else if beliefs.iter().all(|b|
            match b.get() {
                PayoffFuncContainer::Invest(_) => true,
                _ => false,
            }
        ) {
            let beliefs = beliefs.into_iter().map(|b| match b.unpack() {
                PayoffFuncContainer::Invest(payoff_func) => Box::new(payoff_func) as Box<dyn PayoffFunc<InvestActions>>,
                _ => unreachable!(),
            }).collect();
            StateContainer::Invest(Box::new(HetBeliefs::new(beliefs).unwrap()))
        }
        else {
            panic!("All beliefs must be payoff functions of the same action type")
        };
        Self {
            state: state_container,
            class: "HetBeliefs",
        }
    }

    // pub fn belief(&self, i: usize) -> PyPayoffFunc {
    //     unpack_py_enum! {
    //         [StateContainer](state) = self.state.clone();
    //         {
    //             let belief = state.belief(i);
                    //to-do!
    //         }
    //     }
    // }

    #[getter]
    pub fn atype(&self) -> String {
        format!("{}", self.get().object_type())
    }

    fn __str__(&self) -> String {
        format!("State ({}): atype = {}", self.class, self.atype())
    }
}

pub fn as_state(x: &PyAny) -> PyState {
    if let Ok(y) = x.extract::<PyState>() {
        y
    }
    else if let Ok(y) = x.extract::<PyPayoffFunc>() {
        PyState::common_beliefs(y)
    }
    else if let Ok(y) = x.extract::<Vec<PyPayoffFunc>>() {
        PyState::het_beliefs(y)
    }
    else {
        panic!("Expected State, PayoffFunc, or list of PayoffFuncs, got something else");
    }
}
