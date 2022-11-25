use crate::py::*;


#[derive(Clone)]
#[pyclass(name = "Aggregator")]
pub struct PyAggregator(pub BoxedAggregator<Actions>);