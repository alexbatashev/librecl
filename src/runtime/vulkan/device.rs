use super::queue::InOrderQueue;
use crate::common::cl_types::*;
use crate::common::device::Device as CommonDevice;
use crate::common::platform::ClPlatform;
use std::sync::{Arc, Weak};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType, QueueFamily};
use vulkano::device::Queue;
use vulkano::device::{Device as VkDevice, DeviceCreateInfo, Features, QueueCreateInfo};
use vulkano::VulkanObject;

pub struct Device {
    physical_device: PhysicalDevice<'static>,
    logical_device: Arc<VkDevice>,
    device_type: cl_device_type,
    platform: Weak<ClPlatform>,
    queue: Arc<Queue>,
}

impl Device {
    pub fn new(
        platform: Weak<ClPlatform>,
        physical_device: PhysicalDevice<'static>,
        queue_family: QueueFamily,
    ) -> Device {
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
        return Device {
            physical_device,
            logical_device: device.clone(),
            device_type,
            platform,
            queue: queues.next().unwrap(),
        };
    }

    pub fn get_logical_device(&self) -> Arc<VkDevice> {
        self.logical_device.internal_object();
        return self.logical_device.clone();
    }

    pub fn get_queue(&self) -> Arc<Queue> {
        return self.queue.clone();
    }
}

impl CommonDevice for Device {
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
    fn get_platform(&self) -> cl_platform_id {
        return Weak::as_ptr(&self.platform) as *mut ClPlatform;
    }

    fn create_queue(&self, context: cl_context, device: cl_device_id) -> cl_command_queue {
        // TODO support OoO queues;
        return InOrderQueue::new(context, device);
    }
}

// TODO actually implement thread safety
unsafe impl Sync for Device {}
