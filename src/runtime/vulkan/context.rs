use crate::common::cl_types::*;
use crate::common::context::ClContext;
use crate::common::context::Context as CommonContext;
use crate::common::device::ClDevice;
use crate::common::memory::ClMem;
use librecl_compiler::ClangFrontend;
use librecl_compiler::VulkanBackend;
use std::sync::Arc;
use tokio::runtime::Runtime;
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
    memory_objects: Vec<Box<ClMem>>,
    // TODO make per-device
    allocator: Arc<vk_mem::Allocator>,
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

        let device = match unsafe { owned_devices[0].as_ref() }.unwrap() {
            ClDevice::Vulkan(ref device) => device,
            _ => panic!(),
        };

        let entry = unsafe { ash::Entry::load() }.unwrap();
        let ash_instance = unsafe {
            ash::Instance::load(
                entry.static_fn(),
                device.get_logical_device().instance().internal_object(),
            )
        };
        let ash_device = unsafe {
            ash::Device::load(
                ash_instance.fp_v1_0(),
                device.get_logical_device().internal_object(),
            )
        };

        let physical_device = &device.get_physical_device().internal_object();

        let alloc_create_info =
            vk_mem::AllocatorCreateInfo::new(&ash_instance, &ash_device, physical_device);

        let allocator = Arc::new(vk_mem::Allocator::new(alloc_create_info).unwrap());

        let ctx = Box::into_raw(Box::<ClContext>::new(
            Context {
                devices: owned_devices,
                error_callback,
                callback_user_data,
                threading_runtime: runtime,
                clang_fe: ClangFrontend::new(),
                vulkan_be: VulkanBackend::new(),
                memory_objects: vec![],
                allocator,
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

    pub fn get_allocator(&self) -> Arc<vk_mem::Allocator> {
        return self.allocator.clone();
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
    fn create_buffer(&mut self, context: cl_context, size: usize, _flags: cl_mem_flags) -> cl_mem {
        if self.devices.len() == 1 {
            let buffer: ClMem =
                SingleDeviceBuffer::new(self.allocator.clone(), context, size).into();

            return Box::into_raw(Box::new(buffer));
        } else {
            unimplemented!();
        }
    }
}

unsafe impl Sync for Context {}
unsafe impl Send for Context {}
