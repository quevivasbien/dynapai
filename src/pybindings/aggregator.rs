use rayon::prelude::*;
use crate::pycontainer;
use crate::{init_rep, unpack_py_enum_on_strategies};

use crate::py::*;


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

fn extract_init<'a, A, P>(init: &'a PyAny) -> PyResult<InitGuess<A>>
where A: ActionType, P: PyContainer<Item = A> + FromPyObject<'a>
{
    match init.extract::<Vec<P>>() {
        Ok(s) => {
            let actions = s.into_iter().map(|x| x.unpack()).collect();
            Ok(InitGuess::Fixed(Strategies::from_actions(actions)))
        }
        Err(_) => match init.extract::<usize>() {
            Ok(n) => Ok(InitGuess::Random(n)),
            Err(_) => Err(value_error(
                "init must be either a list of actions or a positive integer; maybe you provided the wrong type of actions?"
            ))
        }
    }
}


#[derive(Clone)]
#[pyclass(name = "SolverResult")]
pub struct PySolverResult {
    pub status: String,
    pub strategies: Option<PyStrategies>,
}
pycontainer!(PySolverResult(strategies: Option<PyStrategies>));

#[pymethods]
impl PySolverResult {
    #[new]
    pub fn new(status: String, strategies: Option<PyStrategies>) -> Self {
        Self{ status, strategies }
    }

    pub fn optimum<'py>(&self, py: Python) -> PyResult<Py<PyAny>> {
        match self.get() {
            Some(s) => Ok(match s.clone().unpack() {
                StrategiesContainer::Basic(s) => s.into_py(py),
                StrategiesContainer::Invest(s) => s.into_py(py),
            }),
            None => Err(value_error(format!("no optimum found, status was {}", self.status)))
        }
    }

    pub fn __str__(&self) -> String {
        let s_string = match self.get() {
            Some(s) => s.__str__(),
            None => "None".to_string()
        };
        format!("SolverResult:\nstatus = {}\nstrategies = {}", self.status, s_string)
    }
}

macro_rules! maybe_options {
    ($init:ident : $init_ty:ty, $pyoptions:ident) => {
        {
            let init_guess = match extract_init::<_, $init_ty>($init) {
                Ok(init) => init,
                Err(e) => return PySolverResult::new(format!("Error when processing init: {}", e), None)
            };
            expand_options(init_guess, $pyoptions)
        }
    };
    ($init:ident : vec[$n:expr] $init_ty:ty, $pyoptions:ident) => {
        {
            let init_guess = match extract_init::<_, $init_ty>($init) {
                Ok(init) => init,
                Err(e) => return vec![PySolverResult::new(format!("Error when processing init: {}", e), None); $n]
            };
            expand_options(init_guess, $pyoptions)
        }
    }
}

macro_rules! solve_with {
    ($aggregator:expr, $options:expr, $unpacker:ident) => {
        {
            match solve($aggregator, $options) {
                Ok(strategies) => PySolverResult::new(
                    "success".to_string(),
                    Some(PyStrategies::$unpacker(strategies)),
                ),
                Err(e) => PySolverResult::new(format!("Error when solving: {}", e), None),
            }
        }
    }
}


#[derive(Clone)]
pub enum AggregatorContainer {
    Basic(Box<dyn Aggregator<Actions>>),
    Invest(Box<dyn Aggregator<InvestActions>>),
}

#[derive(Clone)]
#[pyclass(name = "Aggregator")]
pub struct PyAggregator(pub AggregatorContainer);
pycontainer!(PyAggregator(AggregatorContainer));

#[pymethods]
impl PyAggregator {
    #[new]
    #[args(end_on_win = "false")]
    pub fn new(state: &PyAny, gammas: Vec<f64>, end_on_win: bool) -> PyResult<Self> {
        match as_state(state).unpack() {
            StateContainer::Basic(state) => {
                let discounter = match ExponentialDiscounter::new(state, Array::from(gammas)) {
                    Ok(discounter) => discounter,
                    Err(e) => return Err(value_error(format!("When constructing aggregator: {}", e))),
                };
                Ok(Self(AggregatorContainer::Basic(
                    if end_on_win {
                        match EndsOnContestWin::new(discounter) {
                            Ok(aggregator) => Box::new(aggregator),
                            Err(e) => return Err(value_error(format!("When constructing aggregator: {}", e))),
                        }
                    }
                    else {
                        Box::new(discounter)
                    }
                )))
            },
            StateContainer::Invest(state) => {
                let discounter = match InvestExpDiscounter::new(state, Array::from(gammas)) {
                    Ok(discounter) => discounter,
                    Err(e) => return Err(value_error(format!("When constructing aggregator: {}", e))),
                };
                Ok(Self(AggregatorContainer::Invest(
                    if end_on_win {
                        match EndsOnContestWin::new(discounter) {
                            Ok(aggregator) => Box::new(aggregator),
                            Err(e) => return Err(value_error(format!("When constructing aggregator: {}", e))),
                        }
                    }
                    else {
                        Box::new(discounter)
                    }
                )))
            }
        }
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

    pub fn u_i(&self, i: usize, strategies: &PyAny) -> f64 {
        unpack_py_enum_on_strategies! {
            &self.0 => AggregatorContainer(aggregator);
            strategies => strategies;
            aggregator.u_i(i, &strategies)
        }
    }

    pub fn u<'py>(&self, py: Python<'py>, strategies: &PyAny) -> &'py PyArray1<f64> {
        unpack_py_enum_on_strategies! {
            &self.0 => AggregatorContainer(aggregator);
            strategies => strategies;
            aggregator.u(&strategies).into_pyarray(py)
        }
    }

    #[args(options = "&DEFAULT_OPTIONS")]
    pub fn solve(&self, init: &PyAny, options: &PySolverOptions) -> PySolverResult {
        match &self.0 {
            AggregatorContainer::Basic(aggregator)
                => solve_with!(aggregator.as_ref(), &maybe_options!(init: PyActions, options), from_basic),
            AggregatorContainer::Invest(aggregator)
                => solve_with!(aggregator.as_ref(), &maybe_options!(init: PyInvestActions, options), from_invest),
        }
    }
}


#[derive(Clone)]
pub enum ScenarioContainer {
    Basic(Vec<Box<dyn Aggregator<Actions>>>),
    Invest(Vec<Box<dyn Aggregator<InvestActions>>>),
}

#[derive(Clone)]
#[pyclass(name = "Scenario")]
pub struct PyScenario(pub ScenarioContainer);
pycontainer!(PyScenario(ScenarioContainer));

#[pymethods]
impl PyScenario {
    #[new]
    pub fn new(aggregators: Vec<PyAggregator>) -> PyResult<Self> {
        if aggregators.iter().all(|a|
            match a {
                PyAggregator(AggregatorContainer::Basic(_)) => true,
                PyAggregator(AggregatorContainer::Invest(_)) => false
            }
        ) {
            Ok(Self(ScenarioContainer::Basic(
                aggregators.into_iter().map(|a| match a.unpack() {
                    AggregatorContainer::Basic(a) => a,
                    _ => unreachable!()
                }).collect()
            )))
        }
        else if aggregators.iter().all(|a|
            match a {
                PyAggregator(AggregatorContainer::Basic(_)) => false,
                PyAggregator(AggregatorContainer::Invest(_)) => true
            }
        ) {
            Ok(Self(ScenarioContainer::Invest(
                aggregators.into_iter().map(|a| match a.unpack() {
                    AggregatorContainer::Invest(a) => a,
                    _ => unreachable!()
                }).collect()
            )))
        }
        else {
            Err(value_error("All aggregators must be of the same type"))
        }
    }

    #[args(options = "&DEFAULT_OPTIONS")]
    pub fn solve(&self, init: &PyAny, options: &PySolverOptions) -> Vec<PySolverResult> {
        match &self.0 {
            ScenarioContainer::Basic(scenario) => {
                let options = maybe_options!(init: vec[scenario.len()] PyActions, options);
                scenario.par_iter().map(|aggregator|
                    solve_with!(aggregator.as_ref(), &options, from_basic)
                ).collect()
            },
            ScenarioContainer::Invest(scenario) => {
                let options = maybe_options!(init: vec[scenario.len()] PyInvestActions, options);
                scenario.par_iter().map(|aggregator|
                    solve_with!(aggregator.as_ref(), &options, from_invest)
                ).collect()
            }
        }
    }
}