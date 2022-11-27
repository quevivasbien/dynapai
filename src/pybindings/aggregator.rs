use crate::py::*;


#[derive(Clone)]
#[pyclass(name = "Aggregator")]
pub struct PyAggregator(pub Box<dyn Aggregator<Actions>>);