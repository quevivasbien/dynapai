use pyo3::prelude::*;

pub mod aggregator;
pub mod cost_func;
pub mod csf;
pub mod disaster_cost;
pub mod payoff_func;
pub mod prod_func;
pub mod reward_func;
pub mod risk_func;
pub mod solve;
pub mod state;
pub mod strategies;

pub mod pybindings;
pub mod utils;

pub mod prelude;

pub use pybindings as py;

/// A Python module implemented in Rust.
#[pymodule]
fn dynapai(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<py::PyActions>()?;
    m.add_class::<py::PyInvestActions>()?;
    m.add_class::<py::PyProdFunc>()?;
    m.add_class::<py::PyPayoffFunc>()?;
    m.add_class::<py::PyState>()?;
    m.add_class::<py::PySolverOptions>()?;
    m.add_class::<py::PyAggregator>()?;
    Ok(())
}
