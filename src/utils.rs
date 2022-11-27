pub trait CastableAs<T: 'static> {
    fn as_any(&self) -> &dyn std::any::Any;
    fn cast(&self) -> Option<&T> {
        self.as_any().downcast_ref::<T>()
    }
}

#[macro_export]
macro_rules! impl_castable {
    ($from:ty as $to:ty) => {
        impl<$( $param: $($tr)+* ),*> CastableAs<$to> for $from {
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }
    };
}
