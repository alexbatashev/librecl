use super::queue::InOrderQueue;
use crate::api::cl_types::ClObjectImpl;
use crate::api::cl_types::*;
use crate::interface::ContextKind;
use crate::interface::DeviceKind;
use crate::interface::QueueKind;
use crate::interface::{DeviceImpl, PlatformKind};
use crate::sync::{self, SharedPtr, UnsafeHandle, WeakPtr};
use librecl_compiler::Compiler;
use ocl_type_wrapper::ClObjImpl;
use std::sync::Arc;
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType, QueueFamily};
use vulkano::device::Queue;
use vulkano::device::{Device as VkDevice, DeviceCreateInfo, Features, QueueCreateInfo};

#[derive(ClObjImpl, Clone)]
pub struct Device {
    physical_device: PhysicalDevice<'static>,
    logical_device: Arc<VkDevice>,
    device_type: cl_device_type,
    platform: WeakPtr<PlatformKind>,
    queue: Arc<Queue>,
    compiler: Arc<Compiler>,
    #[cl_handle]
    handle: UnsafeHandle<cl_device_id>,
}

impl Device {
    pub fn new(
        platform: WeakPtr<PlatformKind>,
        physical_device: PhysicalDevice<'static>,
        queue_family: QueueFamily,
    ) -> SharedPtr<DeviceKind> {
        // TODO figure out if we need queues at all
        let features = Features {
            shader_int64: true,
            ..Features::none()
        };
        let (device, mut queues) = VkDevice::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
                enabled_features: features,
                ..Default::default()
            },
        )
        .expect("could not create a device");
        // TODO correct device type
        let device_type = match physical_device.properties().device_type {
            PhysicalDeviceType::Cpu => cl_device_type::CPU,
            PhysicalDeviceType::IntegratedGpu => cl_device_type::GPU,
            PhysicalDeviceType::DiscreteGpu => cl_device_type::GPU,
            PhysicalDeviceType::VirtualGpu => cl_device_type::GPU,
            PhysicalDeviceType::Other => cl_device_type::ACC,
        };
        let device = Device {
            physical_device,
            logical_device: device.clone(),
            device_type,
            platform,
            queue: queues.next().unwrap(),
            compiler: Compiler::new(),
            handle: UnsafeHandle::null(),
        }
        .into();
        // This is intentional
        let raw_device = _cl_device_id::wrap(device);
        return DeviceKind::try_from_cl(raw_device).unwrap();
    }

    pub fn get_logical_device(&self) -> Arc<VkDevice> {
        return self.logical_device.clone();
    }

    pub fn get_physical_device(&self) -> PhysicalDevice {
        return self.physical_device.clone();
    }

    pub fn get_queue(&self) -> Arc<Queue> {
        return self.queue.clone();
    }

    pub fn get_compiler(&self) -> Arc<Compiler> {
        self.compiler.clone()
    }
}

impl DeviceImpl for Device {
    fn get_device_type(&self) -> cl_device_type {
        return self.device_type;
    }

    fn get_device_name(&self) -> String {
        return self.physical_device.properties().device_name.clone();
    }

    fn is_available(&self) -> bool {
        // TODO modern laptops allow external GPUs to be connected
        return true;
    }
    fn get_platform(&self) -> WeakPtr<PlatformKind> {
        return self.platform.clone();
    }

    fn create_queue(
        &self,
        context: SharedPtr<ContextKind>,
        device: SharedPtr<DeviceKind>,
    ) -> QueueKind {
        // TODO support OoO queues;
        return InOrderQueue::new(
            SharedPtr::downgrade(&context),
            SharedPtr::downgrade(&device),
        )
        .into();
    }
}

// TODO actually implement thread safety
// unsafe impl Sync for Device {}
