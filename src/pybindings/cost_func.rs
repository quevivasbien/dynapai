use crate::py::*;
use crate::{def_py_enum, unpack_py_enum, pycontainer};


def_py_enum!(CostFuncContainer(Box<dyn CostFunc>));

#[derive(Clone)]
#[pyclass(name = "CostFunc")]
pub struct PyCostFunc {
    pub cost_func: CostFuncContainer,
    pub class: &'static str,
}
pycontainer!(PyCostFunc(cost_func: CostFuncContainer));

#[pymethods]
impl PyCostFunc {
    #[staticmethod]
    pub fn fixed_basic(r: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            cost_func: CostFuncContainer::Basic(
                match BasicFixedCost::new(
                    match Array::from_shape_vec(
                        (r.len(), 2),
                        vec![r; 2].concat()
                    ) {
                        Ok(r) => r,
                        Err(e) => return Err(value_error(format!("{}", e)))
                    }
                ) {
                    Ok(r) => Box::new(r),
                    Err(e) => return Err(value_error(e))
                }
            ),
            class: "FixedCost",
        })
    }

    #[staticmethod]
    pub fn fixed_invest(r: Vec<f64>, r_inv: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            cost_func: CostFuncContainer::Invest(
                match InvestFixedCost::new(
                    match Array::from_shape_vec(
                        (r.len(), 4),
                        vec![vec![r, r_inv].concat(); 2].concat()
                    ) {
                        Ok(r) => r,
                        Err(e) => return Err(value_error(format!("{}", e)))
                    }
                ) {
                    Ok(r) => Box::new(r),
                    Err(e) => return Err(value_error(e))
                }
            ),
            class: "FixedCost",
        })
    }

    #[staticmethod]
    #[args(r_share = "None")]
    pub fn fixed_sharing(r: Vec<f64>, r_inv: Vec<f64>, r_share: Option<Vec<f64>>) -> PyResult<Self> {
        let r_share = match r_share {
            Some(r_share) => r_share,
            None => vec![0.0; r.len()]
        };
        Ok(Self {
            cost_func: CostFuncContainer::Sharing(
                match SharingFixedCost::new(
                    match Array::from_shape_vec(
                        (r.len(), 6),
                        vec![vec![r, r_inv, r_share].concat(); 2].concat()
                    ) {
                        Ok(r) => r,
                        Err(e) => return Err(value_error(format!("{}", e)))
                    }
                ) {
                    Ok(r) => Box::new(r),
                    Err(e) => return Err(value_error(e))
                }
            ),
            class: "FixedCost",
        })
    }
    
    #[staticmethod]
    #[args(r_inv = "None", r_share = "None")]
    pub fn fixed(
        r: Vec<f64>,
        r_inv: Option<Vec<f64>>,
        r_share: Option<Vec<f64>>
    ) -> PyResult<Self> {
        if let Some(r_inv) = r_inv {
            if let Some(r_share) = r_share {
                Self::fixed_sharing(r, r_inv, Some(r_share))
            }
            else {
                Self::fixed_invest(r, r_inv)
            }
        }
        else {
            Self::fixed_basic(r)
        }
    }

    pub fn c_i(&self, i: usize, actions: &PyActions) -> f64 {
        unpack_py_enum! {
            [CostFuncContainer, ActionContainer](cost_func, actions) = self.get(), actions.get(); 
            cost_func.c_i(i, &actions)
        }
    }
    pub fn c<'py>(&self, py: Python<'py>, actions: &PyActions) -> &'py PyArray1<f64> {
        unpack_py_enum! {
            [CostFuncContainer, ActionContainer](cost_func, actions) = self.get(), actions.get(); 
            cost_func.c(&actions).into_pyarray(py)
        }
    }

    pub fn __call__<'py>(&self, py: Python<'py>, actions: &PyActions) -> &'py PyArray1<f64> {
        self.c(py, actions)
    }

    #[getter]
    pub fn atype(&self) -> String {
        format!("{}", self.get().object_type())
    }

    pub fn __str__(&self) -> String {
        format!("CostFunc ({}): atype = {}", self.class, self.atype())
    }

}