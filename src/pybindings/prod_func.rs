use crate::py::*;
use crate::{pycontainer, unpack_py_enum, init_rep};


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
        a_list: Vec<Vec<f64>>, alpha_list: Vec<Vec<f64>>,
        b_list: Vec<Vec<f64>>, beta_list: Vec<Vec<f64>>,
    ) -> Vec<PyProdFunc> {
        init_rep!(PyProdFunc =>
            a = a_list;
            alpha = alpha_list;
            b = b_list;
            beta = beta_list
        )
    }

    fn f_i(&self, i: usize, actions: &PyActions) -> (f64, f64) {
        unpack_py_enum! {
            &actions.0 => ActionContainer(actions);
            self.0.f_i(i, actions)
        }
    }

    fn f<'py>(&self, py: Python<'py>, actions: &PyActions) -> (&'py PyArray1<f64>, &'py PyArray1<f64>) {
        let (s, p) = unpack_py_enum! {
            &actions.0 => ActionContainer(actions);
            self.0.f(actions)
        };
        (s.into_pyarray(py), p.into_pyarray(py))
    }

    fn __call__<'py>(&self, py: Python<'py>, actions: &PyActions) -> (&'py PyArray1<f64>, &'py PyArray1<f64>) {
        self.f(py, actions)
    }

    fn __str__(&self) -> String {
        format!("{}", &self.0)
    }
}
