use rayon::prelude::*;

use crate::py::*;
use crate::{init_rep, unpack_py_enum_on_strategies, pycontainer, def_py_enum, unpack_py_enum, unpack_py_enum_expect};


#[pyclass(name = "SolverOptions")]
pub struct PySolverOptions {
    pub max_iters: u64,
    pub tol: f64,
    pub init_simplex_size: f64,
    pub nm_max_iters: u64,
    pub nm_tol: f64,
}

const DEFAULT_OPTIONS: PySolverOptions = PySolverOptions {
    max_iters: 200,
    tol: 1e-6,
    init_simplex_size: 0.1,
    nm_max_iters: 200,
    nm_tol: 1e-8
};

#[pymethods]
impl PySolverOptions {
    #[new]
    #[args(
        max_iters = "DEFAULT_OPTIONS.max_iters",
        tol = "DEFAULT_OPTIONS.tol",
        init_simplex_size = "DEFAULT_OPTIONS.init_simplex_size",
        nm_max_iters = "DEFAULT_OPTIONS.nm_max_iters",
        nm_tol = "DEFAULT_OPTIONS.nm_tol",
    )]
    fn new(
        max_iters: u64,
        tol: f64,
        init_simplex_size: f64,
        nm_max_iters: u64,
        nm_tol: f64,
    ) -> Self {
        PySolverOptions { max_iters, tol, init_simplex_size, nm_max_iters, nm_tol }
    }

    fn __str__(&self) -> String {
        format!(
            "SolverOptions:\nmax_iters = {}\ntol = {}\ninit_simplex_size = {}\nnm_max_iters = {}\nnm_tol = {}",
            self.max_iters, self.tol, self.init_simplex_size, self.nm_max_iters, self.nm_tol
        )
    }
}


#[derive(Clone)]
#[pyclass(name = "SolverResult")]
pub struct PySolverResult {
    pub status: String,
    pub strategies: Option<Vec<PyActions>>,
}
pycontainer!(PySolverResult(strategies: Option<Vec<PyActions>>));

impl PySolverResult {
    pub fn from_result(res: PyResult<Vec<PyActions>>) -> Self {
        match res {
            Ok(strategies) => PySolverResult {
                status: "success".to_string(),
                strategies: Some(strategies),
            },
            Err(e) => PySolverResult {
                status: format!("Error while solving: {}", e),
                strategies: None,
            }
        }
    }
}

#[pymethods]
impl PySolverResult {
    #[new]
    pub fn new(status: String, strategies: Option<Vec<PyActions>>) -> Self {
        Self{ status, strategies }
    }

    pub fn optimum(&self) -> PyResult<Vec<PyActions>> {
        match self.get() {
            Some(s) => Ok(s.clone()),
            None => Err(value_error(format!("no optimum found, status was {}", self.status)))
        }
    }

    pub fn __str__(&self) -> String {
        let s_string = match self.get() {
            Some(s) => s.iter().enumerate().map(|(t, a)|
                    format!("t = {}, {}", t, a.__str__())
                ).collect::<Vec<_>>().join("\n"),
            None => "None".to_string()
        };
        format!("SolverResult:\nstatus = {}\nstrategies = {}", self.status, s_string)
    }
}

fn expand_options<A: ActionType + Clone>(init_guess: InitGuess<A>, options: &PySolverOptions) -> SolverOptions<A> {
    SolverOptions {
        init_guess: init_guess,
        max_iters: options.max_iters,
        tol: options.tol,
        nm_options: NMOptions {
            init_simplex_size: options.init_simplex_size,
            max_iters: options.nm_max_iters,
            tol: options.nm_tol,
        }
    }
}


macro_rules! maybe_options {
    ($atype:ident, $t:ident, $init:ident, $pyoptions:ident) => {
        {
            let init_guess = if let Some(init) = &$init {
                InitGuess::Fixed(
                    Strategies::from_actions(
                        init.iter().map(|x|
                            if let ActionContainer::$atype(a) = x.get() {
                                a.clone()
                            }
                            else {
                                panic!("wrong action type")
                            }
                        ).collect()
                    )
                )
                
            }
            else if let Some(t) = $t {
                InitGuess::Random(t)
            }
            else {
                return PySolverResult::new(format!("{}", value_error("must provide either init or t")), None);
            };
            expand_options(init_guess, &$pyoptions)
        }
    };
}

macro_rules! solve_with {
    ($aggregator:expr, $options:expr) => {
        {
            match solve($aggregator, $options) {
                Ok(strategies) => Ok(
                    strategies.into_actions().into_iter().map(|a|
                        PyActions::from_data(a.data().clone()).unwrap()
                    ).collect()
                ),
                Err(e) => Err(value_error(format!("Error when solving: {}", e))),
            }
        }
    }
}


def_py_enum!(AggregatorContainer(Box<dyn Aggregator>));

#[derive(Clone)]
#[pyclass(name = "Aggregator")]
pub struct PyAggregator(pub AggregatorContainer);
pycontainer!(PyAggregator(AggregatorContainer));

#[pymethods]
impl PyAggregator {
    #[new]
    #[args(end_on_win = "false")]
    pub fn new(state: &PyAny, gammas: Vec<f64>, end_on_win: bool) -> PyResult<Self> {
        Ok(Self(
            unpack_py_enum! {
                as_state(state).unpack() => StateContainer(state);
                {
                    let discounter = match DynStateDiscounter::new(state, Array::from(gammas)) {
                        Ok(d) => d,
                        Err(e) => return Err(value_error(format!("Error when creating aggregator: {}", e))),
                    };
                    if end_on_win {
                        match EndsOnContestWin::new(discounter) {
                            Ok(a) => Box::new(a),
                            Err(e) => return Err(value_error(format!("Error when creating aggregator: {}", e))),
                        }
                    }
                    else {
                        Box::new(discounter)
                    }
                } => AggregatorContainer
            }
        ))
    }

    #[staticmethod]
    #[args(end_on_win = "false")]
    pub fn expand_from(
        state_list: Vec<&PyAny>,
        gammas_list: Vec<Vec<f64>>,
        end_on_win: bool,
    ) -> Vec<Self> {
        init_rep!(Self =>
            state = state_list;
            gammas = gammas_list;
            end_on_win = vec![end_on_win]
        )
    }

    pub fn u_i(&self, i: usize, strategies: Vec<PyActions>) -> PyResult<f64> {
        let pystrategies = PyStrategies::from_actions_list(strategies)?;
        unpack_py_enum_on_strategies! {
            &self.0 => AggregatorContainer(aggregator);
            pystrategies => strategies;
            Ok(aggregator.u_i(i, &strategies))
        }
    }

    pub fn u<'py>(&self, py: Python<'py>, strategies: Vec<PyActions>) -> PyResult<&'py PyArray1<f64>> {
        let pystrategies = PyStrategies::from_actions_list(strategies)?;
        unpack_py_enum_on_strategies! {
            &self.0 => AggregatorContainer(aggregator);
            pystrategies => strategies;
            Ok(aggregator.u(&strategies).into_pyarray(py))
        }
    }

    #[args(t = "None", init = "None", options = "&DEFAULT_OPTIONS")]
    pub fn solve(&self, t: Option<usize>, init: Option<Vec<PyActions>>, options: &PySolverOptions) -> PySolverResult {
        let res = match &self.0 {
            AggregatorContainer::Basic(aggregator) => {
                let options = maybe_options!(Basic, t, init, options);
                solve_with!(aggregator.as_ref(), &options)
            },
            AggregatorContainer::Invest(aggregator) => {
                let options = maybe_options!(Invest, t, init, options);
                solve_with!(aggregator.as_ref(), &options)
            },
            _ => unimplemented!()
        };
        match res {
            Ok(strategies) => PySolverResult::new("success".to_string(), Some(strategies)),
            Err(e) => PySolverResult::new(format!("{}", e), None),
        }
    }

    #[pyo3(name  = "atype")]
    #[getter]
    pub fn class(&self) -> String {
        format!("{}", self.get().object_type())
    }
}


#[derive(Clone)]
pub enum ScenarioContainer {
    Basic(Vec<Box<dyn Aggregator<Actions>>>),
    Invest(Vec<Box<dyn Aggregator<InvestActions>>>),
    Sharing(Vec<Box<dyn Aggregator<SharingActions>>>),
}

#[derive(Clone)]
#[pyclass(name = "Scenario")]
pub struct PyScenario(pub ScenarioContainer);
pycontainer!(PyScenario(ScenarioContainer));

macro_rules! build_agg_with_type {
    ( $aggregators:expr ; $obj_type:ident ) => {
        {
            let mut aggs_out = Vec::with_capacity($aggregators.len());
            for a in $aggregators.into_iter() {
                let a_ = unpack_py_enum_expect!(
                    a.unpack() => AggregatorContainer::$obj_type(aggregators)
                )?;
                aggs_out.push(a_);
            }
            ScenarioContainer::$obj_type(aggs_out)
        }
    };
}

#[pymethods]
impl PyScenario {
    #[new]
    pub fn new(aggregators: Vec<PyAggregator>) -> PyResult<Self> {
        if aggregators.is_empty() {
            return Err(value_error("must provide at least one aggregator"));
        }
        let obj_type = aggregators[0].get().object_type();
        Ok(Self(
            if obj_type == ObjectType::Basic {
                build_agg_with_type!(aggregators; Basic)
            }
            else if obj_type == ObjectType::Invest {
                build_agg_with_type!(aggregators; Invest)
            }
            else if obj_type == ObjectType::Sharing {
                build_agg_with_type!(aggregators; Sharing)
            }
            else {
                return Err(value_error("unsupported object type"));
            }
        ))
    }

    #[args(t = "None", init = "None", options = "&DEFAULT_OPTIONS")]
    pub fn solve(&self, t: Option<usize>, init: Option<Vec<PyActions>>, options: &PySolverOptions) -> Vec<PySolverResult> {
        match &self.0 {
            ScenarioContainer::Basic(scenario) => {
                scenario.par_iter().map(|aggregator| {
                    let res = solve_with!(
                        aggregator.as_ref(),
                        &maybe_options!(Basic, t, init, options)
                    );
                    PySolverResult::from_result(res)
                }).collect()
            },
            ScenarioContainer::Invest(scenario) => {
                scenario.par_iter().map(|aggregator| {
                    let res = solve_with!(
                        aggregator.as_ref(),
                        &maybe_options!(Invest, t, init, options)
                    );
                    PySolverResult::from_result(res)
                }).collect()
            }
            ScenarioContainer::Sharing(_) => unimplemented!()
        }
    }
}
