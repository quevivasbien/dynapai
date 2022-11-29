use crate::py::*;
use crate::unpack_py;


#[derive(Clone)]
#[pyclass(name = "ProdFunc")]
pub struct PyProdFunc(pub DefaultProd);

#[pymethods]
impl PyProdFunc {
    #[new]
    fn new(
        a: Vec<f64>, alpha: Vec<f64>,
        b: Vec<f64>, beta: Vec<f64>
    ) -> Self {
        Self(DefaultProd::new(
            Array::from(a),
            Array::from(alpha),
            Array::from(b),
            Array::from(beta),
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
