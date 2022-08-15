use crate::api::cl_types::*;
use crate::api::error_handling::{map_invalid_context, ClError};
use crate::interface::{ContextImpl, ContextKind, DeviceKind, MemKind, ProgramKind};
use crate::sync::{self, *};
use ocl_type_wrapper::ClObjImpl;
use tokio::runtime::Runtime;

use super::Program;
use super::ProgramContent;
use super::SingleDeviceBuffer;

#[derive(ClObjImpl)]
pub struct Context {
    devices: Vec<WeakPtr<DeviceKind>>,
    _error_callback: cl_context_callback,
    _callback_user_data: *mut libc::c_void,
    threading_runtime: Runtime,
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
            _error_callback: error_callback,
            _callback_user_data: callback_user_data,
            threading_runtime: runtime,
            handle: UnsafeHandle::null(),
        }
        .into()
    }
}

impl ContextImpl for Context {
    fn notify_error(&self, _message: String) {
        unimplemented!();
    }
    fn has_device(&self, _device: WeakPtr<DeviceKind>) -> bool {
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
    fn create_program_with_spirv(&self, spirv: &[i8]) -> Result<ProgramKind, ClError> {
        let context: SharedPtr<ContextKind> =
            FromCl::try_from_cl(*self.handle.value()).map_err(map_invalid_context)?;
        Ok(Program::new(
            SharedPtr::downgrade(&context),
            ProgramContent::SPIRV(spirv.to_vec()),
        ))
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
