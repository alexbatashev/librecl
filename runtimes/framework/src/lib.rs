use std::ptr::null;

use libc::c_int;
use libc::c_uint;
use std::any::Any;
use std::vec;

mod cl_types;
mod context;
mod device;
mod kernel;
mod platform;
mod program;
mod queue;
mod utils;

pub use cl_types::*;
pub use context::*;
pub use device::*;
pub use kernel::*;
pub use platform::*;
pub use program::*;
pub use queue::*;
pub use utils::*;
