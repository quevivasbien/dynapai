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

pub fn positive_bound(x: f64) -> f64
{
    1. - f64::exp(-x)
}

// macro for creating vector of related structs
#[macro_export]
macro_rules! init_rep {
    ($type:ident => $($field:ident = $val:expr);* ) => {
        {
            let max_n = [$( $val.len() ),*].iter().fold(0 as usize, |acc, l| {
                if *l > acc {*l} else {acc}
            });
            let zipped = itertools::izip!($(
                    if $val.len() < max_n {
                        let mut val = $val.clone();
                        let last = val.last().unwrap().clone();
                        val.extend(std::iter::repeat(last).take(max_n - val.len()));
                        val
                    }
                    else {
                        $val
                    }
            ),*);
            zipped.map(|($($field),*)| {
                $type::new($($field.clone()),*).unwrap()
            }).collect::<Vec<_>>()
        }
    };
}
