#[macro_export]
macro_rules! lcl_contract {
    ($cond:expr, $message:tt, $exitCode:tt) => {
        if !$cond {
            println!("Assertion failed {}: {}", stringify!($cond), $message);
            return 1;
        }
    };
}
