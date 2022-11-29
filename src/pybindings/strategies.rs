use crate::py::*;
use crate::pycontainer;

#[derive(Clone)]
#[pyclass(name = "Actions")]
pub struct PyActions(pub Actions);
pycontainer!(PyActions(Actions));

#[pymethods]
impl PyActions {
    #[new]
    fn new(data: PyReadonlyArray2<f64>) -> Self {
        Self(Actions::from_array(data.as_array().to_owned()))
    }

    #[staticmethod]
    fn from_inputs(xs: Vec<f64>, xp: Vec<f64>) -> Self {
        let data = stack![Axis(1), Array::from(xs), Array::from(xp)];
        Self(Actions::from_array(data))
    }

    fn data<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        PyArray2::from_array(py, self.0.data())
    }

    fn n(&self) -> usize {
        self.0.n()
    }

    fn xs<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        PyArray1::from_array(py, &self.0.xs())
    }

    fn xp<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        PyArray1::from_array(py, &self.0.xp())
    }

    fn __str__(&self) -> String {
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
    fn new(data: PyReadonlyArray2<f64>) -> Self {
        Self(InvestActions::from_array(data.as_array().to_owned()))
    }

    #[staticmethod]
    fn from_inputs(
        xs: Vec<f64>, xp: Vec<f64>,
        inv_s: Vec<f64>, inv_p: Vec<f64>
    ) -> Self {
        let data = stack![Axis(1), Array::from(xs), Array::from(xp), Array::from(inv_s), Array::from(inv_p)];
        Self(InvestActions::from_array(data))
    }

    fn data<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        PyArray2::from_array(py, self.0.data())
    }

    fn n(&self) -> usize {
        self.0.n()
    }

    fn xs<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        PyArray1::from_array(py, &self.0.xs())
    }

    fn xp<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        PyArray1::from_array(py, &self.0.xp())
    }

    fn inv_s<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        PyArray1::from_array(py, &self.0.inv_s())
    }

    fn inv_p<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        PyArray1::from_array(py, &self.0.inv_p())
    }

    fn __str__(&self) -> String {
        format!("{}", &self.0 as &dyn InvestActionType)
    }
}
