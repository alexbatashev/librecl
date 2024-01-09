use super::cl_types::*;
use backtrace::Backtrace;
use core::fmt;

#[derive(Debug, Clone)]
pub struct ErrorDescription {
    pub message: String,
    pub backtrace: Backtrace,
}

impl std::convert::From<&str> for ErrorDescription {
    fn from(message: &str) -> Self {
        ErrorDescription {
            message: message.to_owned(),
            backtrace: Backtrace::new(),
        }
    }
}

impl std::convert::From<String> for ErrorDescription {
    fn from(message: String) -> Self {
        ErrorDescription {
            message,
            backtrace: Backtrace::new(),
        }
    }
}

include!("cl_error_codes.rs");

impl std::convert::Into<Result<(), ClError>> for ClError {
    fn into(self) -> Result<(), ClError> {
        match self {
            ClError::Success(_) => Ok(()),
            _ => Err(self),
        }
    }
}

#[macro_export]
macro_rules! success {
    () => {
        ClError::Success("".into()).into()
    };
}

pub fn map_invalid_context(reason: String) -> ClError {
    ClError::InvalidContext(format!("invalid context: {}", reason).into())
}

pub fn map_invalid_device(reason: String) -> ClError {
    ClError::InvalidDevice(format!("invalid device: {}", reason).into())
}

pub fn map_invalid_kernel(reason: String) -> ClError {
    ClError::InvalidKernel(format!("invalid kernel: {}", reason).into())
}

pub fn map_invalid_queue(reason: String) -> ClError {
    ClError::InvalidCommandQueue(format!("invalid queue: {}", reason).into())
}

pub fn map_invalid_mem(reason: String) -> ClError {
    ClError::InvalidMemObject(format!("invalid mem: {}", reason).into())
}

pub fn map_invalid_event(reason: String) -> ClError {
    ClError::InvalidEvent(format!("invalid event: {}", reason).into())
}

pub fn map_invalid_program(reason: String) -> ClError {
    ClError::InvalidProgram(format!("invalid program: {}", reason).into())
}
