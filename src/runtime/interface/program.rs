use super::{ContextKind, DeviceKind, KernelKind};
use crate::sync::*;
use enum_dispatch::enum_dispatch;

use crate::api::cl_types::{cl_program, ClObjectImpl};
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

// TODO make this safe!!!
unsafe impl Send for ProgramKind {}

/// Common interfaces for Program objects for all backends.
#[enum_dispatch(ProgramKind)]
pub trait ProgramImpl: ClObjectImpl<cl_program> {
    /// Returns associated context.
    fn get_context(&self) -> WeakPtr<ContextKind>;

    // TODO allow options
    /// Compiles program in input state.
    fn compile_program(&mut self, devices: &[WeakPtr<DeviceKind>]) -> bool;
    // TODO allow options and multiple programs
    /// Links programs in built state.
    fn link_programs(&mut self, devices: &[WeakPtr<DeviceKind>]) -> bool;

    /// Creates a new kernel from C function name.
    fn create_kernel(&self, kernel_name: &str) -> KernelKind;
}

impl ClObjectImpl<cl_program> for ProgramKind {
    fn get_cl_handle(&self) -> cl_program {
        match self {
            #[cfg(feature = "vulkan")]
            ProgramKind::Vulkan(prog) => ClObjectImpl::<cl_program>::get_cl_handle(prog),
            #[cfg(feature = "metal")]
            ProgramKind::Metal(prog) => ClObjectImpl::<cl_program>::get_cl_handle(prog),
        }
    }
    fn set_cl_handle(&mut self, handle: cl_program) {
        match self {
            #[cfg(feature = "vulkan")]
            ProgramKind::Vulkan(prog) => ClObjectImpl::<cl_program>::set_cl_handle(prog, handle),
            #[cfg(feature = "metal")]
            ProgramKind::Metal(prog) => ClObjectImpl::<cl_program>::set_cl_handle(prog, handle),
        }
    }
}
