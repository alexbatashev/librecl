#[macro_export]
macro_rules! format_error {
    ($api_name_full:tt, $message:tt, $exit_code:tt) => {
        (|| -> String {
            let api_name = $api_name_full.split("::").last().unwrap_or("unknown");
            let bt = backtrace::Backtrace::new();
            let frame_id = (|bt: &backtrace::Backtrace| -> Option<usize> {
                for (id, f) in bt.frames().into_iter().enumerate().rev() {
                    let frame_id = f.symbols().into_iter().find_map(|s| {
                        if s.name()
                            .unwrap()
                            .as_str()
                            .unwrap_or("")
                            .starts_with(api_name)
                        {
                            return Some(id);
                        }
                        return Option::None;
                    });
                    if frame_id.is_some() {
                        return Some(frame_id.unwrap());
                    }
                }
                return Option::None;
            })(&bt);

            let mut symbol = Option::None;

            if frame_id.is_some() {
                symbol = Some(&bt.frames()[frame_id.unwrap() + 1].symbols()[0]);
            }

            if symbol.is_some() && symbol.unwrap().filename().is_some() {
                let final_message = std::format!(
                    "{}(...) -> {}: called from {} +{} ({}):\n{}\nBacktrace:\n{:?}\n",
                    api_name,
                    $exit_code,
                    symbol
                        .as_ref()
                        .unwrap()
                        .filename()
                        .unwrap_or(std::path::Path::new("unknown_file"))
                        .to_str()
                        .unwrap_or("unknown_file"),
                    symbol.as_ref().unwrap().lineno().unwrap_or(0),
                    symbol
                        .as_ref()
                        .unwrap()
                        .name()
                        .unwrap()
                        .as_str()
                        .unwrap_or("unknown function"),
                    $message,
                    bt
                );

                return final_message;
            }

            return std::format!(
                "{}(...) -> {}\n{}\nBacktrace:\n{:?}",
                api_name,
                $exit_code,
                $message,
                bt
            );
        })()
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
            println!("{}", full_message);
            return $exit_code;
        }
    };
    ($cond:expr, $message:tt, $exit_code:tt, $ret_err:tt) => {
        if !$cond {
            let name = stdext::function_name!();
            let assertion_message =
                std::format!("Assertion {} failed: {}", stringify!($cond), $message);
            let full_message = format_error!(name, assertion_message, $exit_code);
            println!("{}", full_message);
            let ret_err_safe = unsafe { $ret_err.as_ref() };
            if ret_err_safe.is_some() {
                unsafe {
                    *$ret_err = $exit_code;
                }
            }
            return std::ptr::null_mut();
        }
    };
    ($ctx:expr, $cond:expr, $message:tt, $exit_code:tt, $ret_err:tt) => {
        if !$cond {
            let name = stdext::function_name!();
            let assertion_message =
                std::format!("Assertion {} failed: {}", stringify!($cond), $message);
            let full_message = format_error!(name, assertion_message, $exit_code);
            $ctx.notify_error(full_message);
            let ret_err_safe = unsafe { $ret_err.as_ref() };
            if ret_err_safe.is_some() {
                unsafe {
                    *$ret_err = $exit_code;
                }
            }
            return std::ptr::null_mut();
        }
    };
}

#[macro_export]
macro_rules! set_info_str {
    ($str:tt, $ptr:tt, $size_ptr:tt) => {
        let ptr_safe = unsafe { $ptr.as_ref() };
        let size_ptr_safe = unsafe { $size_ptr.as_ref() };

        if !size_ptr_safe.is_none() {
            unsafe {
                *$size_ptr = $str.len() + 1;
            }
        }

        if !ptr_safe.is_none() {
            unsafe {
                libc::strncpy(
                    $ptr as *mut libc::c_char,
                    $str.as_bytes().as_ptr() as *const libc::c_char,
                    $str.len() as libc::size_t,
                );
                *($ptr as *mut u8).offset($str.len() as isize) = 0;
            }
        }
    };
}
