use super::cl_types::ClErrorCode;
use backtrace::Backtrace;

pub struct ClError<'a> {
  pub error_code: ClErrorCode<'a>,
  pub error_message: String,
  pub backtrace: Backtrace,
}

impl<'a> ClError<'a> {
  pub fn new(error_code: ClErrorCode<'a>, error_message: String) -> ClError {
    return ClError{
      error_code,
      error_message,
      backtrace: Backtrace::new(),
    };
  }
}

#[macro_export]
macro_rules! return_error {
    ($maybe_err:tt) => {
        if $maybe_err.is_err() {
          let err = $maybe_err.err().unwrap();
          println!("{}\n{}\nBacktrace:\n{:?}", err.error_code, err.error_message, err.backtrace);

          return err.error_code.value;
        }
    };
}
