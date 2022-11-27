pub fn isapprox(a: f64, b: f64, rtol: f64, atol: f64) -> bool
{
    let maxval = f64::max(a.abs(), b.abs());
    (a - b).abs() <= f64::max(atol, rtol * maxval)
}

pub fn isapprox_iters<I, J>(a: I, b: J, rtol: f64, atol: f64) -> bool
where I: Iterator<Item = f64>, J: Iterator<Item = f64>
{
    a.zip(b).all(|(a, b)| isapprox(a, b, rtol, atol))
}

// macro for creating vector of related structs
#[macro_export]
macro_rules! init_rep {
    ($type:ident => $($field:ident: $fty:ty = $val:expr);* ) => {
        {
            struct ParamReps {
                $($field: Vec<$fty>),*
            }
            impl ParamReps {
                fn new($($field: Vec<$fty>),*) -> Self {
                    let mut out = ParamReps { $($field),* };
                    let max_n = [$(out.$field.len()),*].iter().fold(0 as usize, |acc, l| {
                        if *l > acc {*l} else {acc}
                    });
                    $(
                        if out.$field.len() < max_n {
                            let last = out.$field.last().unwrap().clone();
                            out.$field.extend(std::iter::repeat(last).take(max_n - out.$field.len()));
                        }
                    )*
                    out
                }

                fn expand(&self) -> Vec<$type> {
                    let zipped = itertools::izip!($(&self.$field),*);
                    zipped.map(|($($field),*)| {
                        $type::new($($field.clone()),*).unwrap()
                    }).collect()
                }
            }

            ParamReps::new(
                $($val),*
            ).expand()
        }
    };
}
