use crate::{py::*, pycontainer};

#[derive(Clone)]
#[pyclass(name = "CSF")]
pub struct PyCSF {
    pub csf: Box<dyn CSF>,
    pub class: &'static str,
}
pycontainer!(PyCSF(csf: Box<dyn CSF>));

#[pymethods]
impl PyCSF {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    #[staticmethod]
    pub fn default() -> Self {
        Self{ csf: Box::new(DefaultCSF), class: "Default" }
    }

    #[staticmethod]
    #[args(scale = "1.0")]
    pub fn maybe_no_win(scale: f64) -> PyResult<Self> {
        match MaybeNoWinCSF::new(scale) {
            Ok(csf) => Ok(Self{ csf: Box::new(csf), class: "MaybeNoWin" }),
            Err(e) => Err(value_error(e)),
        }
    }

    pub fn q_i(&self, i: usize, p: Vec<f64>) -> f64 {
        self.csf.q_i(i, Array::from(p).view())
    }

    pub fn q<'py>(&self, py: Python<'py>, p: Vec<f64>) -> &'py PyArray1<f64> {
        self.csf.q(Array::from(p).view()).into_pyarray(py)
    }

    pub fn __str__(&self) -> String {
        format!("CSF ({})", self.class)
    }
}