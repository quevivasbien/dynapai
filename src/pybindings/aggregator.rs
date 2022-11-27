use crate::py::*;

#[derive(Clone)]
#[pyclass(name = "State")]
pub struct PyState(pub Box<dyn State<Actions>>);

fn as_state(x: &PyAny) -> PyState {
    if let Ok(y) = x.extract::<PyState>() {
        y
    }
    else if let Ok(y) = x.extract::<PyPayoffFunc>() {
        PyState(Box::new(CommonBeliefs(Box::new(y.0))))
    }
    else {
        panic!("Expected State or PayoffFunc, got something else");
    }
}

#[derive(Clone)]
#[pyclass(name = "Aggregator")]
pub struct PyAggregator(pub Box<dyn Aggregator<Actions>>);

#[pymethods]
impl PyAggregator {
    #[new]
    #[args(end_on_win = "false")]
    fn new(state: &PyAny, gammas: PyReadonlyArray1<f64>, end_on_win: bool) -> Self {
        let state = as_state(state);
        let discounter = match ExponentialDiscounter::new(state.0, gammas.as_array().to_owned()) {
            Ok(discounter) => discounter,
            Err(e) => panic!("Error when constructing aggregator: {}", e),
        };
        if end_on_win {
            // todo: fix casting from Box<dyn ...>
            // EndsOnContestWin doesn't work rn because of that
            Self(Box::new(EndsOnContestWin::new(discounter).unwrap()))
        }
        else {
            Self(Box::new(discounter))
        }
    }

    fn u_i(&self, i: usize, strategies: Vec<PyActions>) -> f64 {
        let strategies = Strategies::from_actions(strategies.into_iter().map(|x| x.0).collect());
        self.0.u_i(i, &strategies)
    }

    fn u<'py>(&self, py: Python<'py>, strategies: Vec<PyActions>) -> &'py PyArray1<f64> {
        let strategies = Strategies::from_actions(strategies.into_iter().map(|x| x.0).collect());
        self.0.u(&strategies).into_pyarray(py)
    }
}
