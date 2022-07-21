#[macro_export]
macro_rules! format_error {
    ($api_name:tt, $message:tt, $exit_code:tt) => {
        (|| -> &str {
            let bt = backtrace::Backtrace::new();
            let symbol = (|bt: &backtrace::Backtrace| -> Option<backtrace::BacktraceSymbol> {
                for f in bt.frames().into_iter().rev() {
                    let symbol = f.symbols().into_iter().find(|&s| {
                        s.name()
                            .unwrap()
                            .as_str()
                            .unwrap_or("")
                            .starts_with($api_name)
                    });
                    if symbol.is_some() {
                        return Some(symbol.unwrap().clone());
                    }
                }
                return Option::None;
            })(&bt);

            if symbol.is_some() {
                std::format!(
                    "{} -> {}: called from {} +{}:\n{}\nBacktrace:\n{:?}\n",
                    $api_name,
                    $exit_code,
                    symbol
                        .as_ref()
                        .unwrap()
                        .filename()
                        .unwrap_or(std::path::Path::new("unknown_file"))
                        .to_str()
                        .unwrap_or("unknown_file"),
                    symbol.as_ref().unwrap().lineno().unwrap_or(0),
                    $message,
                    bt
                );
            }

            return "";
        })();
    };
}
#[macro_export]
macro_rules! lcl_contract {
    ($cond:expr, $message:tt, $exit_code:tt) => {
        if !$cond {
            let name = stdext::function_name!();
            let assertion_message =
                std::format!("Assertion {} failed: {}", stringify!($cond), $message);
            let full_message = format_error!(name, assertion_message, $exit_code);
            print!("{}", full_message);
            return $exit_code;
        }
    };
}

#[macro_export]
macro_rules! set_info_str {
    ($str:tt, $ptr:tt, $size_ptr:tt) => {};
}
