use crate::common::cl_types::*;
use crate::common::context::ClContext;
use crate::common::context::Context as CommonContext;
use crate::common::memory::ClMem;
use librecl_compiler::ClangFrontend;
use librecl_compiler::VulkanBackend;
use tokio::runtime::Runtime;
use vulkano::device::DeviceOwned;
use vulkano::VulkanObject;

use super::Program;
use super::ProgramContent;
use super::SingleDeviceBuffer;

pub struct Context {
    devices: Vec<cl_device_id>,
    error_callback: cl_context_callback,
    callback_user_data: *mut libc::c_void,
    threading_runtime: Runtime,
    clang_fe: ClangFrontend,
    vulkan_be: VulkanBackend,
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
                vulkan_be: VulkanBackend::new(),
            }
            .into(),
        ));
        return ctx;
    }

    // TODO return with locks?
    pub fn get_clang_fe(&self) -> &ClangFrontend {
        return &self.clang_fe;
    }

    // TODO return with locks?
    pub fn get_vulkan_be(&self) -> &VulkanBackend {
        return &self.vulkan_be;
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
        return Box::leak(Box::new(
            Program::new(context, ProgramContent::Source(source)).into(),
        ));
    }
    fn get_associated_devices(&self) -> &[cl_device_id] {
        return &self.devices.as_slice();
    }

    fn get_threading_runtime(&self) -> &Runtime {
        return &self.threading_runtime;
    }
    fn create_buffer(&self, context: cl_context, size: usize, _flags: cl_mem_flags) -> cl_mem {
        if self.devices.len() == 1 {
            let buffer: ClMem = SingleDeviceBuffer::new(context, size).into();
            let bb = Box::new(buffer);
            std::mem::forget(&bb);
            let mem: *mut ClMem = Box::into_raw(bb);

            match unsafe { mem.as_ref() }.unwrap() {
                ClMem::VulkanSDBuffer(ref b) => b.get_buffer().device().internal_object(),
            };

            return mem;
        } else {
            unimplemented!();
        }
    }
}

unsafe impl Sync for Context {}
unsafe impl Send for Context {}
