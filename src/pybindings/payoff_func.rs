use crate::init_rep;
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

fn make_basic(
    prod_func: Box<DefaultProd>,
    risk_func: Box<dyn RiskFunc>,
    csf: Box<dyn CSF>,
    reward_func: Box<dyn RewardFunc>,
    disaster_cost: Box<dyn DisasterCost>,
    r: Vec<f64>,
) -> Result<PayoffFuncContainer, &'static str> {
    let cost_func = Box::new(
        FixedUnitCost { r: Array::from(r) }
    );
    let pfunc = ModularPayoff::new(
        prod_func,
        risk_func,
        csf,
        reward_func,
        disaster_cost,
        cost_func,
    )?;
    Ok(PayoffFuncContainer::Basic(pfunc))
}

fn make_invest(
    prod_func: Box<DefaultProd>,
    risk_func: Box<dyn RiskFunc>,
    csf: Box<dyn CSF>,
    reward_func: Box<dyn RewardFunc>,
    disaster_cost: Box<dyn DisasterCost>,
    r: Vec<f64>,
    r_inv: Vec<f64>,
) -> Result<PayoffFuncContainer, &'static str> {
    let cost_func = Box::new(
        FixedInvestCost::new(
            Array::from(r),
            Array::from(r_inv),
        )?
    );
    let pfunc = ModularPayoff::new(
        prod_func,
        risk_func,
        csf,
        reward_func,
        disaster_cost,
        cost_func,
    )?;
    Ok(PayoffFuncContainer::Basic(pfunc))
}

#[pymethods]
impl PyPayoffFunc {
    #[new]
    #[args(csf = "None", r_inv = "None")]
    fn new(
        prod_func: PyProdFunc,
        risk_func: PyRiskFunc,
        csf: Option<PyCSF>,
        reward_func: Option<PyRewardFunc>,
        d: Vec<f64>,
        r: Vec<f64>,
        r_inv: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let n = ProdFunc::<Actions>::n(&prod_func.0);
        let prod_func = Box::new(prod_func.unpack());
        let risk_func = risk_func.unpack();
        let csf = match csf {
            None => Box::new(DefaultCSF),
            Some(csf) => csf.unpack(),
        };
        let reward_func = match reward_func {
            None => Box::new(LinearReward::default(n)),
            Some(reward_func) => reward_func.unpack(),
        };
        let disaster_cost = Box::new(ConstantDisasterCost { d: Array::from(d) });
        let pfunc = if let Some(r_inv) = r_inv {
            make_invest(
                prod_func,
                risk_func,
                csf,
                reward_func,
                disaster_cost,
                r,
                r_inv,
            )
        } else {
            make_basic(
                prod_func,
                risk_func,
                csf,
                reward_func,
                disaster_cost,
                r,
            )
        };
        match pfunc {
            Ok(pfunc) => Ok(PyPayoffFunc(pfunc)),
            Err(e) => Err(value_error(e)),
        }
    }

    #[staticmethod]
    pub fn expand_from(
        prod_func_list: Vec<PyProdFunc>,
        risk_func_list: Vec<PyRiskFunc>,
        csf_list: Option<Vec<PyCSF>>,
        reward_func_list: Option<Vec<PyRewardFunc>>,
        d_list: Vec<Vec<f64>>,
        r_list: Vec<Vec<f64>>,
        r_inv_list: Option<Vec<Vec<f64>>>,
    ) -> Vec<PyPayoffFunc> {
        let csf_list = match csf_list {
            None => vec![None],
            Some(csf_list) => csf_list.into_iter().map(|csf| Some(csf)).collect(),
        };
        let reward_func_list = match reward_func_list {
            None => vec![None],
            Some(reward_func_list) => reward_func_list.into_iter().map(|reward_func| Some(reward_func)).collect(),
        };
        let r_inv_list = match r_inv_list {
            None => vec![None],
            Some(r_inv_list) => r_inv_list.into_iter().map(|r_inv| Some(r_inv)).collect(),
        };

        init_rep!(PyPayoffFunc =>
            prod_func = prod_func_list;
            risk_func = risk_func_list;
            csf = csf_list;
            reward_func = reward_func_list;
            d = d_list;
            r = r_list;
            r_inv = r_inv_list
        )
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
