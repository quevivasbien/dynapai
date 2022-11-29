use pyo3::exceptions::PyException;
use pyo3::types::PyList;
use crate::unpack_py_on_strategies;

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
            Err(_) => Err(PyException::new_err("init must be either a list of actions or a positive integer; maybe you provided the wrong type of actions?"))
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

#[pymethods]
impl PyAggregator {
    #[new]
    #[args(end_on_win = "false")]
    fn new(state: &PyAny, gammas: Vec<f64>, end_on_win: bool) -> Self {
        match as_state(state).unpack() {
            StateContainer::Basic(state) => {
                let discounter = match ExponentialDiscounter::new(state, Array::from(gammas)) {
                    Ok(discounter) => discounter,
                    Err(e) => panic!("Error when constructing aggregator: {}", e),
                };
                Self(AggregatorContainer::Basic(
                    if end_on_win {
                        Box::new(EndsOnContestWin::new(discounter).unwrap())
                    }
                    else {
                        Box::new(discounter)
                    }
                ))
            },
            StateContainer::Invest(state) => {
                let discounter = match InvestExpDiscounter::new(state, Array::from(gammas)) {
                    Ok(discounter) => discounter,
                    Err(e) => panic!("Error when constructing aggregator: {}", e),
                };
                Self(AggregatorContainer::Invest(
                    if end_on_win {
                        Box::new(EndsOnContestWin::new(discounter).unwrap())
                    }
                    else {
                        Box::new(discounter)
                    }
                ))
            }
        }
    }

    fn u_i(&self, i: usize, strategies: &PyAny) -> f64 {
        unpack_py_on_strategies! {
            &self.0 => aggregator: AggregatorContainer;
            strategies => strategies;
            aggregator.u_i(i, &strategies)
        }
    }

    fn u<'py>(&self, py: Python<'py>, strategies: &PyAny) -> &'py PyArray1<f64> {
        unpack_py_on_strategies! {
            &self.0 => aggregator: AggregatorContainer;
            strategies => strategies;
            aggregator.u(&strategies).into_pyarray(py)
        }
    }

    #[args(options = "&DEFAULT_OPTIONS")]
    fn solve<'py>(&self, py: Python<'py>, init: &PyAny, options: &PySolverOptions) -> PyResult<&'py PyList> {
        match &self.0 {
            AggregatorContainer::Basic(aggregator) => {
                let init_guess = extract_init::<_, PyActions>(init)?;
                let options = expand_options(init_guess, options);
                let res = solve(aggregator.as_ref(), &options);
                match res {
                    Ok(strategies) => Ok(PyList::new(
                        py,
                        strategies.into_actions().into_iter().map(|x| PyActions(x).into_py(py))
                    )),
                    Err(e) => Err(PyException::new_err(format!("Error when solving: {}", e)))
                }
            },
            AggregatorContainer::Invest(aggregator) => {
                let init_guess = extract_init::<_, PyInvestActions>(init)?;
                let options = expand_options(init_guess, options);
                let res = solve(aggregator.as_ref(), &options);
                match res {
                    Ok(strategies) => Ok(PyList::new(
                        py,
                        strategies.into_actions().into_iter().map(|x| PyInvestActions(x).into_py(py))
                    )),
                    Err(e) => Err(PyException::new_err(format!("Error when solving: {}", e)))
                }
            }
        }
    }
}
