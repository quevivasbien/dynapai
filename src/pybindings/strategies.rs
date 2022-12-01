use crate::py::*;
use crate::pycontainer;
use crate::unpack_py;

#[derive(Clone)]
#[pyclass(name = "Actions")]
pub struct PyActions(pub Actions);
pycontainer!(PyActions(Actions));

#[pymethods]
impl PyActions {
    #[new]
    pub fn new(data: PyReadonlyArray2<f64>) -> Self {
        Self(Actions::from_array(data.as_array().to_owned()))
    }

    #[staticmethod]
    pub fn from_inputs(xs: Vec<f64>, xp: Vec<f64>) -> Self {
        let data = stack![Axis(1), Array::from(xs), Array::from(xp)];
        Self(Actions::from_array(data))
    }

    pub fn data<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        PyArray2::from_array(py, self.0.data())
    }

    pub fn n(&self) -> usize {
        self.0.n()
    }

    pub fn xs<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        PyArray1::from_array(py, &self.0.xs())
    }

    pub fn xp<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        PyArray1::from_array(py, &self.0.xp())
    }

    pub fn __str__(&self) -> String {
        format!("{}", &self.0 as &dyn ActionType)
    }
}

#[derive(Clone)]
#[pyclass(name = "InvestActions")]
pub struct PyInvestActions(pub InvestActions);
pycontainer!(PyInvestActions(InvestActions));

#[pymethods]
impl PyInvestActions {
    #[new]
    pub fn new(data: PyReadonlyArray2<f64>) -> Self {
        Self(InvestActions::from_array(data.as_array().to_owned()))
    }

    #[staticmethod]
    pub fn from_inputs(
        xs: Vec<f64>, xp: Vec<f64>,
        inv_s: Vec<f64>, inv_p: Vec<f64>
    ) -> Self {
        let data = stack![Axis(1), Array::from(xs), Array::from(xp), Array::from(inv_s), Array::from(inv_p)];
        Self(InvestActions::from_array(data))
    }

    pub fn data<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        PyArray2::from_array(py, self.0.data())
    }

    pub fn n(&self) -> usize {
        self.0.n()
    }

    pub fn xs<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        PyArray1::from_array(py, &self.0.xs())
    }

    pub fn xp<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        PyArray1::from_array(py, &self.0.xp())
    }

    pub fn inv_s<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        PyArray1::from_array(py, &self.0.inv_s())
    }

    pub fn inv_p<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        PyArray1::from_array(py, &self.0.inv_p())
    }

    pub fn __str__(&self) -> String {
        format!("{}", &self.0 as &dyn InvestActionType)
    }
}


#[derive(Clone)]
pub enum StrategiesContainer {
    Basic(Vec<PyActions>),
    Invest(Vec<PyInvestActions>),
}

#[derive(Clone)]
#[pyclass(name = "Strategies")]
pub struct PyStrategies(pub StrategiesContainer);
pycontainer!(PyStrategies(StrategiesContainer));

impl PyStrategies {
    pub fn from_basic(strategies: Strategies<Actions>) -> Self {
        Self(StrategiesContainer::Basic(
            strategies.into_actions().into_iter().map(PyActions).collect()
        ))
    }
    pub fn from_invest(strategies: Strategies<InvestActions>) -> Self {
        Self(StrategiesContainer::Invest(
            strategies.into_actions().into_iter().map(PyInvestActions).collect()
        ))
    }
}

#[pymethods]
impl PyStrategies {
    #[new]
    pub fn new(actions_list: &PyAny) -> Self {
        Self(
            unpack_py! {
                actions_list => actions_vec [Vec<PyActions> | Vec<PyInvestActions>];
                actions_vec => StrategiesContainer
            }
        )
    }

    pub fn __str__(&self) -> String {
        let s_string = match &self.0 {
            StrategiesContainer::Basic(actions) => {
                actions.iter().enumerate().map(|(t, a)|
                    format!("t = {}, {}", t, a.__str__())
                ).collect::<Vec<_>>().join("\n")
            },
            StrategiesContainer::Invest(actions) => {
                actions.iter().enumerate().map(|(t, a)|
                    format!("t = {}, {}", t, a.__str__())
                ).collect::<Vec<_>>().join("\n")
            },
        };
        format!("Strategies:\n{}", s_string)
    }
}