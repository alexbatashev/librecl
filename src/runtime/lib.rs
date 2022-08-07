pub mod api;

#[cfg(feature = "vulkan")]
pub mod vulkan;

#[cfg(feature = "metal")]
pub mod metal;

mod cpu;

#[cfg(test)]
mod mock;

pub mod interface;

pub mod sync {
    pub use crate::api::utils::SharedPtr;
    pub use crate::api::utils::UnsafeHandle;
    pub use crate::api::utils::WeakPtr;
}
