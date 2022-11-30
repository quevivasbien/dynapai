use crate::{py::*, pycontainer};

#[derive(Clone)]
pub enum StateContainer {
    Basic(Box<dyn State<Actions>>),
    Invest(Box<dyn State<InvestActions>>),
}

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
        let state_container = match payoff_func.unpack() {
            PayoffFuncContainer::Basic(payoff_func) => {
                StateContainer::Basic(Box::new(CommonBeliefs(Box::new(payoff_func))))
            },
            PayoffFuncContainer::Invest(payoff_func) => {
                StateContainer::Invest(Box::new(CommonBeliefs(Box::new(payoff_func))))
            },
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
                PayoffFuncContainer::Invest(_) => false,
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
                PayoffFuncContainer::Basic(_) => false,
                PayoffFuncContainer::Invest(_) => true,
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

    fn __str__(&self) -> String {
        format!("State ({})", self.class)
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
