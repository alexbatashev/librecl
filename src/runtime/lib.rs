mod api;
pub mod common;

#[cfg(feature = "vulkan")]
pub mod vulkan;

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(test)]
mod mock;

pub mod interface;

pub use crate::common::platform::Platform;
pub use crate::api::cl_types;

pub mod c_cl {
    pub use crate::common::cl_types::*;
    pub use crate::common::context::clCreateContext;
    pub use crate::common::device::clGetDeviceIDs;
    pub use crate::common::platform::clGetPlatformIDs;
    pub use crate::common::platform::clGetPlatformInfo;
}
