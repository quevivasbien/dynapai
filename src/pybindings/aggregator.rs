use rayon::prelude::*;

use crate::py::*;
use crate::{init_rep, pycontainer, def_py_enum, unpack_py_enum, unpack_py_enum_expect};


#[pyclass(name = "SolverOptions")]
pub struct PySolverOptions {
    pub iters: u64,
    pub tol: f64,
    pub init_simplex_size: f64,
    pub nm_iters: u64,
    pub nm_tol: f64,
    pub hist_size: usize,
    pub mixed_samples: usize,
    pub parallel: bool,
}

const DEFAULT_OPTIONS: PySolverOptions = PySolverOptions {
    iters: 200,
    tol: 1e-6,
    init_simplex_size: 0.1,
    nm_iters: 200,
    nm_tol: 1e-8,
    hist_size: 10,
    mixed_samples: 100,
    parallel: true,
};

#[pymethods]
impl PySolverOptions {
    #[new]
    #[args(
        iters = "DEFAULT_OPTIONS.iters",
        tol = "DEFAULT_OPTIONS.tol",
        init_simplex_size = "DEFAULT_OPTIONS.init_simplex_size",
        nm_iters = "DEFAULT_OPTIONS.nm_iters",
        nm_tol = "DEFAULT_OPTIONS.nm_tol",
        hist_size = "DEFAULT_OPTIONS.hist_size",
        mixed_samples = "DEFAULT_OPTIONS.mixed_samples",
        parallel = "DEFAULT_OPTIONS.parallel"
    )]
    fn new(
        iters: u64,
        tol: f64,
        init_simplex_size: f64,
        nm_iters: u64,
        nm_tol: f64,
        hist_size: usize,
        mixed_samples: usize,
        parallel: bool,
    ) -> Self {
        PySolverOptions { iters, tol, init_simplex_size, nm_iters, nm_tol, hist_size, mixed_samples, parallel }
    }

    fn __str__(&self) -> String {
        format!(
            "SolverOptions:\niters = {}\ntol = {}\ninit_simplex_size = {}\nnm_iters = {}\nnm_tol = {}",
            self.iters, self.tol, self.init_simplex_size, self.nm_iters, self.nm_tol
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

    #[getter]
    pub fn optimum(&self) -> PyResult<Vec<PyActions>> {
        match self.get() {
            Some(s) => Ok(s.clone()),
            None => Err(value_error(format!("no optimum found, status was {}", self.status)))
        }
    }

    // todo: add method to get more detailed information about sigma, s, p, payoffs, etc.

    pub fn __str__(&self) -> String {
        let s_string = match self.get() {
            Some(s) => s.iter().enumerate().map(|(t, a)|
                    format!("t = {}, {}", t, a.__str__())
                ).collect::<Vec<_>>().join("\n"),
            None => "None".to_string()
        };
        format!("SolverResult:\nstatus: {}\nstrategies:\n{}", self.status, s_string)
    }
}

fn expand_options<A: ActionType + Clone>(init_guess: InitGuess<A>, options: &PySolverOptions) -> SolverOptions<A> {
    SolverOptions {
        init_guess: init_guess,
        iters: options.iters,
        tol: options.tol,
        nm_options: NMOptions {
            init_simplex_size: options.init_simplex_size,
            iters: options.nm_iters,
            tol: options.nm_tol,
        },
        hist_size: options.hist_size,
        mixed_samples: options.mixed_samples,
        parallel: options.parallel,
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

fn downcast_to_eocwin<A: ActionType + Clone + 'static>(x: &Box<dyn Aggregator<A>>) -> Option<&aggregator::EndsOnContestWin<A, aggregator::DynStateDiscounter<A>>> {
    x.downcast_ref::<EndsOnContestWin<A, DynStateDiscounter<A>>>()
}

def_py_enum!(AggregatorContainer(Box<dyn Aggregator>));

#[derive(Clone)]
#[pyclass(name = "Aggregator")]
pub struct PyAggregator {
    pub aggregator: AggregatorContainer,
    pub end_on_win: bool,
}
pycontainer!(PyAggregator(aggregator: AggregatorContainer));

#[pymethods]
impl PyAggregator {
    #[new]
    #[args(end_on_win = "false")]
    pub fn new(state: &PyAny, gammas: Vec<f64>, end_on_win: bool) -> PyResult<Self> {
        Ok(Self {
            aggregator: unpack_py_enum! {
                [StateContainer](state) = as_state(state).unpack();
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
            },
            end_on_win
        })
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
        unpack_py_enum! {
            [AggregatorContainer, StrategyContainer](aggregator, strategies) = self.get(), pystrategies.get();
            Ok(aggregator.u_i(i, &strategies))
        }
    }

    pub fn u<'py>(&self, py: Python<'py>, strategies: Vec<PyActions>) -> PyResult<&'py PyArray1<f64>> {
        let pystrategies = PyStrategies::from_actions_list(strategies)?;
        unpack_py_enum! {
            [AggregatorContainer, StrategyContainer](aggregator, strategies) = self.get(), pystrategies.get();
            Ok(aggregator.u(&strategies).into_pyarray(py))
        }
    }

    pub fn probas<'py>(&self, py: Python<'py>, strategies: Vec<PyActions>) -> PyResult<&'py PyArray2<f64>> {
        if !self.end_on_win {
            return Err(PyErr::new::<PyTypeError, _>("Can only calculate probas if end_on_win == true"));
        }
        let pystrategies = PyStrategies::from_actions_list(strategies)?;
        unpack_py_enum! {
            [AggregatorContainer, StrategyContainer](agg_box, strategies) = self.get(), pystrategies.get();
            {
                let aggregator = downcast_to_eocwin(agg_box).unwrap();
                Ok(aggregator.probas(&strategies).into_pyarray(py))
            }
        }
    }

    #[args(t = "None", init = "None", options = "&DEFAULT_OPTIONS")]
    pub fn solve(&self, t: Option<usize>, init: Option<Vec<PyActions>>, options: &PySolverOptions) -> PySolverResult {
        let res = match self.get() {
            AggregatorContainer::Basic(aggregator) => {
                let options = maybe_options!(Basic, t, init, options);
                solve_with!(aggregator.as_ref(), &options)
            },
            AggregatorContainer::Invest(aggregator) => {
                let options = maybe_options!(Invest, t, init, options);
                solve_with!(aggregator.as_ref(), &options)
            },
            AggregatorContainer::Sharing(aggregator) => {
                let options = maybe_options!(Sharing, t, init, options);
                solve_with!(aggregator.as_ref(), &options)
            },
        };
        match res {
            Ok(strategies) => PySolverResult::new("success".to_string(), Some(strategies)),
            Err(e) => PySolverResult::new(format!("{}", e), None),
        }
    }

    pub fn state0(&self) -> PyState {
        PyState {
            state: unpack_py_enum! {
                [AggregatorContainer](aggregator) = self.get();
                aggregator.state0().clone() => StateContainer
            },
            class: "?"
        }
    }

    pub fn states(&self, strategies: Vec<PyActions>) -> Vec<PyState> {
        let mut states = Vec::with_capacity(strategies.len());
        states.push(self.state0());
        for actions in strategies[0..strategies.len() - 1].iter() {
            let last_state = states.last().unwrap();
            states.push(
                PyState {
                    state: unpack_py_enum! {
                        [AggregatorContainer, ActionContainer, StateContainer](aggregator, actions, state) = self.get(), actions.get(), last_state.get();
                        {
                            let mut state = state.clone();
                            aggregator.advance_state(&mut state, actions);
                            state
                        } => StateContainer
                    },
                    class: "?"
                }
            );
        }
        states
    }

    #[getter]
    pub fn atype(&self) -> String {
        format!("{}", self.get().object_type())
    }

    pub fn __str__(&self) -> String {
        format!("Aggregator: atype = {}, end_on_win = {}", self.atype(), self.end_on_win)
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
                    a.unpack() => AggregatorContainer::$obj_type
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
        match self.get() {
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
            ScenarioContainer::Sharing(scenario) => {
                scenario.par_iter().map(|aggregator| {
                    let res = solve_with!(
                        aggregator.as_ref(),
                        &maybe_options!(Sharing, t, init, options)
                    );
                    PySolverResult::from_result(res)
                }).collect()
            }
        }
    }
}
