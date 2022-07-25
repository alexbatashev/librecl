use crate::common::cl_types::*;
use crate::common::context::ClContext;
use crate::common::context::Context as CommonContext;
use librecl_compiler::ClangFrontend;
use tokio::runtime::Runtime;

pub struct Context {
    devices: Vec<cl_device_id>,
    error_callback: cl_context_callback,
    callback_user_data: *mut libc::c_void,
    threading_runtime: Runtime,
    clang_fe: ClangFrontend,
}

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
        let ctx = Box::into_raw(Box::<ClContext>::new(
            Context {
                devices: owned_devices,
                error_callback,
                callback_user_data,
                threading_runtime: runtime,
                clang_fe: ClangFrontend::new(),
            }
            .into(),
        ));
        return ctx;
    }
}

impl CommonContext for Context {
    fn notify_error(&self, message: String) {
        unimplemented!();
    }
    fn has_device(&self, device: cl_device_id) -> bool {
        return self.devices.contains(&device);
    }
    fn create_program_with_source(&self, source: String) -> cl_program {
        unimplemented!();
    }
    fn get_associated_devices(&self) -> &[cl_device_id] {
        unimplemented!();
    }

    fn get_threading_runtime(&self) -> &Runtime {
        return &self.threading_runtime;
    }
    fn create_buffer(&self, size: usize, flags: cl_mem_flags) -> cl_mem {
        unimplemented!();
    }
}
