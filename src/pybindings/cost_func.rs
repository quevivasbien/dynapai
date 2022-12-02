use crate::py::*;
use crate::{def_py_enum, pycontainer, unpack_py_enum_on_actions};


def_py_enum!(CostFuncContainer(Box<dyn CostFunc>));

#[derive(Clone)]
#[pyclass(name = "CostFunc")]
pub struct PyCostFunc(pub CostFuncContainer);
pycontainer!(PyCostFunc(CostFuncContainer));

#[pymethods]
impl PyCostFunc {
    #[new]
    pub fn new(r: Vec<f64>) -> Self {
        Self::fixed(r)
    }
    #[staticmethod]
    pub fn fixed(r: Vec<f64>) -> Self {
        let cost_func = Box::new(
            FixedUnitCost { r: Array::from(r) }
        );
        PyCostFunc(CostFuncContainer::Basic(cost_func))
    }
    #[staticmethod]
    pub fn fixed_invest(r: Vec<f64>, r_inv: Vec<f64>) -> PyResult<Self> {
        let cost_func = Box::new(
            match FixedInvestCost::new(
                Array::from(r),
                Array::from(r_inv),
            ) {
                Ok(cost_func) => cost_func,
                Err(e) => return Err(value_error(e)),
            }
        );
        Ok(PyCostFunc(CostFuncContainer::Invest(cost_func)))
    }

    pub fn c_i(&self, i: usize, actions: &PyActions) -> f64 {
        unpack_py_enum_on_actions! {
            self.get() => CostFuncContainer(cost_func);
            actions => actions;
            cost_func.c_i(i, &actions)
        }
    }
    pub fn c<'py>(&self, py: Python<'py>, actions: &PyActions) -> &'py PyArray1<f64> {
        unpack_py_enum_on_actions! {
            self.get() => CostFuncContainer(cost_func);
            actions => actions;
            cost_func.c(&actions).into_pyarray(py)
        }
    }

    #[pyo3(name  = "atype")]
    #[getter]
    pub fn class(&self) -> String {
        format!("{}", self.get().object_type())
    }
}