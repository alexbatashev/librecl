use crate::common::cl_types::*;
use crate::common::context::Context as CommonContext;
use librecl_compiler::ClangFrontend;
use librecl_compiler::MetalBackend;
use tokio::runtime::Runtime;

use super::Program;
use super::ProgramContent;
use super::SingleDeviceBuffer;

pub struct Context {
    devices: Vec<cl_device_id>,
    error_callback: cl_context_callback,
    callback_user_data: *mut libc::c_void,
    threading_runtime: Runtime,
    clang_fe: ClangFrontend,
    metal_be: MetalBackend,
}

unsafe impl Sync for Context {}
unsafe impl Send for Context {}

impl Context {
    pub fn new(
        devices: &[cl_device_id],
        error_callback: cl_context_callback,
        callback_user_data: *mut libc::c_void,
    ) -> cl_context {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(3)
            .build()
            .unwrap();
        let mut owned_devices = vec![];
        owned_devices.extend_from_slice(devices);

        return Box::into_raw(Box::new(
            Context {
                devices: owned_devices,
                error_callback,
                callback_user_data,
                threading_runtime: runtime,
                clang_fe: ClangFrontend::new(),
                metal_be: MetalBackend::new(),
            }
            .into(),
        ));
    }

    pub fn get_clang_fe(&self) -> &ClangFrontend {
        return &self.clang_fe;
    }
    pub fn get_metal_be(&self) -> &MetalBackend {
        return &self.metal_be;
    }
}

impl CommonContext for Context {
    fn notify_error(&self, message: String) {
        unimplemented!();
    }
    fn has_device(&self, device: cl_device_id) -> bool {
        return self.devices.contains(&device);
    }
    fn create_program_with_source(&self, context: cl_context, source: String) -> cl_program {
        return Program::new(context, ProgramContent::Source(source));
    }
    fn create_buffer(&mut self, context: cl_context, size: usize, _flags: cl_mem_flags) -> cl_mem {
        // TODO support multiple devices in one context
        return SingleDeviceBuffer::new(context, size);
    }
    fn get_threading_runtime(&self) -> &Runtime {
        unimplemented!();
    }
    fn get_associated_devices(&self) -> &[cl_device_id] {
        return &self.devices;
    }
}
