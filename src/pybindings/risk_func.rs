use crate::{py::*, pycontainer};

#[derive(Clone)]
#[pyclass(name = "RiskFunc")]
pub struct PyRiskFunc {
    pub risk_func: Box<dyn RiskFunc>,
    pub class: &'static str,
}
pycontainer!(PyRiskFunc(risk_func: Box<dyn RiskFunc>));

#[pymethods]
impl PyRiskFunc {

    #[staticmethod]
    pub fn winner_only(theta: Vec<f64>) -> Self {
        Self{ risk_func: Box::new(WinnerOnlyRisk { theta: Array::from(theta) }), class: "WinnerOnly" }
    }

    pub fn sigma_i(&self, i: usize, s: Vec<f64>, p: Vec<f64>) -> f64 {
        self.risk_func.sigma_i(i, Array::from(s).view(), Array::from(p).view())
    }

    pub fn sigma<'py>(&self, py: Python<'py>, s: Vec<f64>, p: Vec<f64>) -> &'py PyArray1<f64> {
        self.risk_func.sigma(Array::from(s).view(), Array::from(p).view()).into_pyarray(py)
    }

    pub fn __str__(&self) -> String {
        format!("RiskFunc ({})", self.class)
    }
}
