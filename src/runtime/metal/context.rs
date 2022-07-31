use crate::api::cl_types::*;
use crate::interface::{ContextImpl, ContextKind, DeviceKind, MemKind, ProgramKind};
use crate::sync::{self, *};
use librecl_compiler::ClangFrontend;
use librecl_compiler::MetalBackend;
use ocl_type_wrapper::ClObjImpl;
use tokio::runtime::Runtime;

use super::Program;
use super::ProgramContent;
use super::SingleDeviceBuffer;

#[derive(ClObjImpl)]
pub struct Context {
    devices: Vec<WeakPtr<DeviceKind>>,
    error_callback: cl_context_callback,
    callback_user_data: *mut libc::c_void,
    threading_runtime: Runtime,
    clang_fe: ClangFrontend,
    metal_be: MetalBackend,
    handle: UnsafeHandle<cl_context>,
}

unsafe impl Sync for Context {}
unsafe impl Send for Context {}

impl Context {
    pub fn new(
        devices: &[WeakPtr<DeviceKind>],
        error_callback: cl_context_callback,
        callback_user_data: *mut libc::c_void,
    ) -> ContextKind {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(3)
            .build()
            .unwrap();
        let mut owned_devices = vec![];
        owned_devices.extend_from_slice(devices);

        Context {
            devices: owned_devices,
            error_callback,
            callback_user_data,
            threading_runtime: runtime,
            clang_fe: ClangFrontend::new(),
            metal_be: MetalBackend::new(),
            handle: UnsafeHandle::null(),
        }
        .into()
    }

    pub fn get_clang_fe(&self) -> &ClangFrontend {
        return &self.clang_fe;
    }
    pub fn get_metal_be(&self) -> &MetalBackend {
        return &self.metal_be;
    }
}

impl ContextImpl for Context {
    fn notify_error(&self, message: String) {
        unimplemented!();
    }
    fn has_device(&self, device: WeakPtr<DeviceKind>) -> bool {
        return true;
        // TODO actual comparison
        // return self.devices.contains(&device);
    }
    fn create_program_with_source(&self, source: String) -> ProgramKind {
        let context: SharedPtr<ContextKind> = FromCl::try_from_cl(*self.handle.value()).unwrap();
        return Program::new(
            SharedPtr::downgrade(&context),
            ProgramContent::Source(source),
        );
    }
    fn create_buffer(&mut self, size: usize, _flags: cl_mem_flags) -> MemKind {
        let context: SharedPtr<ContextKind> = FromCl::try_from_cl(*self.handle.value()).unwrap();
        // TODO support multiple devices in one context
        return SingleDeviceBuffer::new(SharedPtr::downgrade(&context), size);
    }
    fn get_threading_runtime(&self) -> &Runtime {
        return &self.threading_runtime;
    }
    fn get_associated_devices(&self) -> &[WeakPtr<DeviceKind>] {
        return &self.devices;
    }
}
