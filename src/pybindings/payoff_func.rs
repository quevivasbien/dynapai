use crate::cost_func::*;
use crate::csf::DefaultCSF;
use crate::disaster_cost::ConstantDisasterCost;
use crate::payoff_func::*;
use crate::prod_func::*;
use crate::reward_func::LinearReward;
use crate::risk_func::WinnerOnlyRisk;
use crate::strategies::*;

use crate::py::*;

#[derive(Clone)]
#[pyclass(name = "PayoffFunc")]
pub struct PyPayoffFunc(pub BoxedModularPayoff<Actions>);

#[pymethods]
impl PyPayoffFunc {
    #[new]
    fn new(
        prod_func: PyProdFunc,
        theta: PyReadonlyArray1<f64>,
        d: PyReadonlyArray1<f64>,
        r: PyReadonlyArray1<f64>,
    ) -> Self {
        let n = ProdFunc::<Actions>::n(&prod_func.0);
        Self(BoxedModularPayoff::new(
            Box::new(prod_func.0),
            Box::new(WinnerOnlyRisk { theta: theta.as_array().to_owned() }),
            Box::new(DefaultCSF),
            Box::new(LinearReward::default(n)),
            Box::new(ConstantDisasterCost { d: d.as_array().to_owned() }),
            Box::new(FixedUnitCost { r: r.as_array().to_owned() }),
        ).expect("invalid payoff function parameters"))
    }

    fn u_i(&self, i: usize, actions: &PyActions) -> f64 {
        self.0.u_i(i, &actions.0)
    }

    fn u<'py>(&self, py: Python<'py>, actions: &PyActions) -> &'py PyArray1<f64> {
        self.0.u(&actions.0).into_pyarray(py)
    }

    fn __call__<'py>(&self, py: Python<'py>, actions: &PyActions) -> &'py PyArray1<f64> {
        self.u(py, actions)
    }
}

#[derive(Clone)]
#[pyclass(name = "InvestPayoffFunc")]
pub struct PyInvestPayoffFunc(pub BoxedModularPayoff<InvestActions>);

#[pymethods]
impl PyInvestPayoffFunc {
    #[new]
    fn new(
        prod_func: PyProdFunc,
        theta: PyReadonlyArray1<f64>,
        d: PyReadonlyArray1<f64>,
        r: PyReadonlyArray1<f64>,
        r_inv: PyReadonlyArray1<f64>,
    ) -> Self {
        let n = ProdFunc::<InvestActions>::n(&prod_func.0);
        Self(BoxedModularPayoff::new(
            Box::new(prod_func.0),
            Box::new(WinnerOnlyRisk { theta: theta.as_array().to_owned() }),
            Box::new(DefaultCSF),
            Box::new(LinearReward::default(n)),
            Box::new(ConstantDisasterCost { d: d.as_array().to_owned() }),
            Box::new(FixedInvestCost::new(r.as_array().to_owned(), r_inv.as_array().to_owned())),
        ).expect("invalid payoff function parameters"))
    }

    fn u_i(&self, i: usize, actions: &PyInvestActions) -> f64 {
        self.0.u_i(i, &actions.0)
    }

    fn u<'py>(&self, py: Python<'py>, actions: &PyInvestActions) -> &'py PyArray1<f64> {
        self.0.u(&actions.0).into_pyarray(py)
    }

    fn __call__<'py>(&self, py: Python<'py>, actions: &PyInvestActions) -> &'py PyArray1<f64> {
        self.u(py, actions)
    }
}