use enum_dispatch::enum_dispatch;
use std::rc::Weak;
use super::{DeviceKind, ContextKind, KernelKind};

#[cfg(feature = "vulkan")]
use crate::vulkan::Program as VkProgram;

#[cfg(feature = "metal")]
use crate::metal::Program as MTLProgram;

#[enum_dispatch]
#[repr(C)]
pub enum ProgramKind {
    #[cfg(feature = "vulkan")]
    Vulkan(VkProgram),
    #[cfg(feature = "metal")]
    Metal(MTLProgram),
}

/// Common interfaces for Program objects for all backends.
#[enum_dispatch(ProgramKind)]
pub trait ProgramImpl {
    /// Returns associated context.
    fn get_context(&self) -> Weak<ContextKind>;

    // TODO allow options
    /// Compiles program in input state.
    fn compile_program(&mut self, devices: &[Weak<DeviceKind>]) -> bool;
    // TODO allow options and multiple programs
    /// Links programs in built state.
    fn link_programs(&mut self, devices: &[Weak<DeviceKind>]) -> bool;

    fn create_kernel(&self, kernel_name: &str) -> KernelKind;
}
