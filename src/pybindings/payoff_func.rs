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
        theta: PyReadonlyArray1<f64>,
        d: PyReadonlyArray1<f64>,
        r: PyReadonlyArray1<f64>,
        r_inv: Option<PyReadonlyArray1<f64>>,
    ) -> Self {
        let n = ProdFunc::<Actions>::n(&prod_func.0);
        let prod_func = Box::new(prod_func.0);
        let risk_func = Box::new(WinnerOnlyRisk { theta: theta.as_array().to_owned() });
        let csf = Box::new(DefaultCSF);
        let reward_func = Box::new(LinearReward::default(n));
        let disaster_cost = Box::new(ConstantDisasterCost { d: d.as_array().to_owned() });
        if let Some(r_inv) = r_inv {
            let cost_func = Box::new(FixedInvestCost::new(
                r.as_array().to_owned(),
                r_inv.as_array().to_owned(),
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
            let cost_func = Box::new(FixedUnitCost { r: r.as_array().to_owned() });
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
