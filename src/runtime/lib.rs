pub mod common;

#[cfg(feature = "vulkan")]
pub mod vulkan;

#[cfg(feature = "metal")]
pub mod metal;

pub mod c_cl {
    pub use crate::common::cl_types::*;
    pub use crate::common::platform::clGetPlatformIDs;
    pub use crate::common::platform::clGetPlatformInfo;
}

pub use crate::common::platform::Platform;
