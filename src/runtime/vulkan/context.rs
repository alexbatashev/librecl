use crate::api::cl_types::*;
use crate::api::error_handling::ClError;
use crate::interface::ContextImpl;
use crate::interface::ContextKind;
use crate::interface::DeviceKind;
use crate::interface::MemKind;
use crate::interface::ProgramKind;
use crate::sync::{self, SharedPtr, UnsafeHandle, WeakPtr};
use ocl_type_wrapper::ClObjImpl;
use std::ops::Deref;
use std::sync::Arc;
use tokio::runtime::Runtime;
use vulkano::VulkanObject;

use super::Program;
use super::ProgramContent;
use super::SingleDeviceBuffer;

#[derive(ClObjImpl)]
pub struct Context {
    devices: Vec<WeakPtr<DeviceKind>>,
    _error_callback: cl_context_callback,
    _callback_user_data: *mut libc::c_void,
    threading_runtime: Runtime,
    // TODO make per-device
    allocator: Arc<vk_mem::Allocator>,
    #[cl_handle]
    handle: UnsafeHandle<cl_context>,
}

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
        owned_devices.extend_from_slice(&devices);

        let owned_device = owned_devices[0].upgrade().unwrap();
        let device = match owned_device.deref() {
            DeviceKind::Vulkan(device) => device,
            #[allow(unreachable_patterns)]
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

        Context {
            devices: owned_devices,
            _error_callback: error_callback,
            _callback_user_data: callback_user_data,
            threading_runtime: runtime,
            allocator,
            handle: UnsafeHandle::null(),
        }
        .into()
    }

    pub fn get_allocator(&self) -> Arc<vk_mem::Allocator> {
        return self.allocator.clone();
    }
}

impl ContextImpl for Context {
    fn notify_error(&self, _message: String) {
        unimplemented!();
    }
    fn has_device(&self, _device: WeakPtr<DeviceKind>) -> bool {
        // FIXME find out how to check for device existance.
        return true;
        // return self.devices.contains(&device);
    }
    fn create_program_with_source(&self, source: String) -> ProgramKind {
        let context: SharedPtr<ContextKind> = FromCl::try_from_cl(*self.handle.value()).unwrap();
        Program::new(
            SharedPtr::downgrade(&context),
            ProgramContent::Source(source),
        )
        .into()
    }

    fn create_program_with_spirv(&self, spirv: &[i8]) -> Result<ProgramKind, ClError> {
        let context: SharedPtr<ContextKind> =
            FromCl::try_from_cl(*self.handle.value()).map_err(|_err| {
                ClError::new(
                    ClErrorCode::InvalidContext,
                    "failed to acquire owning reference to context".to_owned(),
                )
            })?;

        Ok(Program::new(
            SharedPtr::downgrade(&context),
            ProgramContent::SPIRV(spirv.to_vec()),
        )
        .into())
    }

    fn get_associated_devices(&self) -> &[WeakPtr<DeviceKind>] {
        return &self.devices.as_slice();
    }

    fn get_threading_runtime(&self) -> &Runtime {
        return &self.threading_runtime;
    }
    fn create_buffer(&mut self, size: usize, _flags: cl_mem_flags) -> MemKind {
        let owned_context = FromCl::try_from_cl(self.get_cl_handle()).unwrap();
        if self.devices.len() == 1 {
            let buffer = SingleDeviceBuffer::new(
                self.allocator.clone(),
                SharedPtr::downgrade(&owned_context),
                size,
            )
            .into();
            return buffer;
        } else {
            unimplemented!();
        }
    }
}

unsafe impl Sync for Context {}
unsafe impl Send for Context {}
