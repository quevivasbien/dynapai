use crate::py::*;
use crate::unpack_py;

#[derive(Clone)]
#[pyclass(name = "ProdFunc")]
pub struct PyProdFunc(pub DefaultProd);

#[pymethods]
impl PyProdFunc {
    #[new]
    fn new(
        a: PyReadonlyArray1<f64>, alpha: PyReadonlyArray1<f64>,
        b: PyReadonlyArray1<f64>, beta: PyReadonlyArray1<f64>
    ) -> Self {
        Self(DefaultProd::new(
            a.as_array().to_owned(),
            alpha.as_array().to_owned(),
            b.as_array().to_owned(),
            beta.as_array().to_owned(),
        ).expect("invalid production function parameters"))
    }

    fn f_i(&self, i: usize, actions: &PyAny) -> (f64, f64) {
        unpack_py! {
            actions => actions [PyActions | PyInvestActions];
            self.0.f_i(i, &actions.0)
        }
    }

    fn f<'py>(&self, py: Python<'py>, actions: &PyAny) -> (&'py PyArray1<f64>, &'py PyArray1<f64>) {
        let (s, p) = unpack_py! {
            actions => actions [PyActions | PyInvestActions];
            self.0.f(&actions.0)
        };
        (s.into_pyarray(py), p.into_pyarray(py))
    }

    fn __call__<'py>(&self, py: Python<'py>, actions: &PyAny) -> (&'py PyArray1<f64>, &'py PyArray1<f64>) {
        self.f(py, actions)
    }

    fn __str__(&self) -> String {
        format!("{}", &self.0)
    }
}
