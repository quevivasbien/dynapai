use crate::py::*;
use crate::{pycontainer, unpack_py_enum_on_actions};


#[derive(Clone)]
pub enum CostFuncContainer {
    Basic(Box<dyn CostFunc<Actions>>),
    Invest(Box<dyn CostFunc<InvestActions>>),
}

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

    pub fn c_i(&self, i: usize, actions: &PyAny) -> f64 {
        unpack_py_enum_on_actions! {
            &self.0 => CostFuncContainer(cfunc);
            actions => actions;
            cfunc.c_i(i, &actions)
        }
    }
    pub fn c<'py>(&self, py: Python<'py>, actions: &PyAny) -> &'py PyArray1<f64> {
        unpack_py_enum_on_actions! {
            &self.0 => CostFuncContainer(cfunc);
            actions => actions;
            cfunc.c(&actions).into_pyarray(py)
        }
    }
}