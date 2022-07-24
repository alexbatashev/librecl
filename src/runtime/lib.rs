pub mod common;

#[cfg(feature = "vulkan")]
pub mod vulkan;

#[cfg(feature = "metal")]
pub mod metal;

pub use crate::common::platform::Platform;

pub mod c_cl {
    pub use crate::common::cl_types::*;
    pub use crate::common::context::clCreateContext;
    pub use crate::common::device::clGetDeviceIDs;
    pub use crate::common::platform::clGetPlatformIDs;
    pub use crate::common::platform::clGetPlatformInfo;
}
