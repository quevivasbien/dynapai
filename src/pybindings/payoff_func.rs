use crate::py::*;
use crate::pycontainer;
use crate::unpack_py_on_actions;

#[derive(Clone)]
pub enum PayoffFuncContainer {
    Basic(ModularPayoff<Actions>),
    Invest(ModularPayoff<InvestActions>),
}

#[derive(Clone)]
#[pyclass(name = "PayoffFunc")]
pub struct PyPayoffFunc(pub PayoffFuncContainer);
pycontainer!(PyPayoffFunc(PayoffFuncContainer));

#[pymethods]
impl PyPayoffFunc {
    #[new]
    #[args(r_inv = "None")]
    fn new(
        prod_func: PyProdFunc,
        theta: Vec<f64>,
        d: Vec<f64>,
        r: Vec<f64>,
        r_inv: Option<Vec<f64>>,
    ) -> Self {
        let n = ProdFunc::<Actions>::n(&prod_func.0);
        let prod_func = Box::new(prod_func.0);
        let risk_func = Box::new(WinnerOnlyRisk { theta: Array::from(theta) });
        let csf = Box::new(DefaultCSF);
        let reward_func = Box::new(LinearReward::default(n));
        let disaster_cost = Box::new(ConstantDisasterCost { d: Array::from(d) });
        if let Some(r_inv) = r_inv {
            let cost_func = Box::new(FixedInvestCost::new(
                Array::from(r),
                Array::from(r_inv),
            ));
            Self(PayoffFuncContainer::Invest(ModularPayoff::new(
                prod_func,
                risk_func,
                csf,
                reward_func,
                disaster_cost,
                cost_func,
            ).expect("invalid payoff function parameters")))
        } else {
            let cost_func = Box::new(FixedUnitCost { r: Array::from(r) });
            Self(PayoffFuncContainer::Basic(ModularPayoff::new(
                prod_func,
                risk_func,
                csf,
                reward_func,
                disaster_cost,
                cost_func
            ).expect("invalid payoff function parameters")))
        }
    }

    fn u_i(&self, i: usize, actions: &PyAny) -> f64 {
        unpack_py_on_actions! {
            &self.0 => pfunc: PayoffFuncContainer;
            actions => actions;
            pfunc.u_i(i, &actions)
        }
    }

    fn u<'py>(&self, py: Python<'py>, actions: &PyAny) -> &'py PyArray1<f64> {
        unpack_py_on_actions! {
            &self.0 => pfunc: PayoffFuncContainer;
            actions => actions;
            pfunc.u(&actions).into_pyarray(py)
        }
    }

    fn __call__<'py>(&self, py: Python<'py>, actions: &PyAny) -> &'py PyArray1<f64> {
        self.u(py, actions)
    }
}
