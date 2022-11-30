use crate::{py::*, pycontainer};
use crate::{unpack_py, init_rep};


#[derive(Clone)]
#[pyclass(name = "ProdFunc")]
pub struct PyProdFunc(pub DefaultProd);
pycontainer!(PyProdFunc(DefaultProd));

#[pymethods]
impl PyProdFunc {
    #[new]
    fn new(
        a: Vec<f64>, alpha: Vec<f64>,
        b: Vec<f64>, beta: Vec<f64>
    ) -> PyResult<Self> {
        let prod_func = DefaultProd::new(
            Array::from(a),
            Array::from(alpha),
            Array::from(b),
            Array::from(beta),
        );
        match prod_func {
            Ok(p) => Ok(Self(p)),
            Err(e) => Err(value_error(e)),
        }
    }

    #[staticmethod]
    fn expand_from(
        a: Vec<Vec<f64>>, alpha: Vec<Vec<f64>>,
        b: Vec<Vec<f64>>, beta: Vec<Vec<f64>>,
    ) -> Vec<PyProdFunc> {
        init_rep!(PyProdFunc =>
            a: Vec<f64> = a;
            alpha: Vec<f64> = alpha;
            b: Vec<f64> = b;
            beta: Vec<f64> = beta
        )
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
