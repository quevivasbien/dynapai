use crate::init_rep;
use crate::py::*;
use crate::pycontainer;
use crate::unpack_py_enum;
use crate::unpack_py_enum_on_actions;

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
    #[args(csf = "None", reward_func = "None", r_inv = "None")]
    pub fn new(
        prod_func: PyProdFunc,
        risk_func: PyRiskFunc,
        d: Vec<f64>,
        cost_func: PyCostFunc,
        csf: Option<PyCSF>,
        reward_func: Option<PyRewardFunc>,
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
        Ok(Self(unpack_py_enum! {
            cost_func.unpack() => CostFuncContainer(cost_func);
            match ModularPayoff::new(
                prod_func,
                risk_func,
                csf,
                reward_func,
                disaster_cost,
                cost_func,
            ) {
                Ok(pfunc) => pfunc,
                Err(e) => return Err(value_error(e)),
            } => PayoffFuncContainer
        }))
    }

    #[staticmethod]
    pub fn expand_from(
        prod_func_list: Vec<PyProdFunc>,
        risk_func_list: Vec<PyRiskFunc>,
        d_list: Vec<Vec<f64>>,
        cost_func_list: Vec<PyCostFunc>,
        csf_list: Option<Vec<PyCSF>>,
        reward_func_list: Option<Vec<PyRewardFunc>>,
    ) -> Vec<PyPayoffFunc> {
        let csf_list = match csf_list {
            None => vec![None],
            Some(csf_list) => csf_list.into_iter().map(|csf| Some(csf)).collect(),
        };
        let reward_func_list = match reward_func_list {
            None => vec![None],
            Some(reward_func_list) => reward_func_list.into_iter().map(|reward_func| Some(reward_func)).collect(),
        };

        init_rep!(PyPayoffFunc =>
            prod_func = prod_func_list;
            risk_func = risk_func_list;
            d = d_list;
            cost_func = cost_func_list;
            csf = csf_list;
            reward_func = reward_func_list
        )
    }

    pub fn u_i(&self, i: usize, actions: &PyAny) -> f64 {
        unpack_py_enum_on_actions! {
            &self.0 => PayoffFuncContainer(pfunc);
            actions => actions;
            pfunc.u_i(i, &actions)
        }
    }

    pub fn u<'py>(&self, py: Python<'py>, actions: &PyAny) -> &'py PyArray1<f64> {
        unpack_py_enum_on_actions! {
            &self.0 => PayoffFuncContainer(pfunc);
            actions => actions;
            pfunc.u(&actions).into_pyarray(py)
        }
    }

    pub fn __call__<'py>(&self, py: Python<'py>, actions: &PyAny) -> &'py PyArray1<f64> {
        self.u(py, actions)
    }
}
