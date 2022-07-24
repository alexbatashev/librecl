use crate::common::cl_types::*;
use crate::common::device::ClDevice;
use crate::common::device::Device as CommonDevice;
use crate::common::platform::ClPlatform;
use std::sync::{Arc, Weak};
use vulkano::device::physical::{PhysicalDevice, QueueFamily};
use vulkano::device::{Device as VkDevice, DeviceCreateInfo, QueueCreateInfo};
use super::queue::InOrderQueue;

pub struct Device {
    physical_device: PhysicalDevice<'static>,
    logical_device: Arc<VkDevice>,
    device_type: cl_device_type,
    platform: Weak<ClPlatform>,
}

impl Device {
    pub fn new(platform: Weak<ClPlatform>, physical_device: PhysicalDevice<'static>, queue_family: QueueFamily) -> Device {
        // TODO figure out if we need queues at all
        let (device, _) = VkDevice::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
                ..Default::default()
            },
        )
        .expect("could not create a device");
        // TODO correct device type
        return Device {
            physical_device,
            logical_device: device,
            device_type: cl_device_type::GPU,
            platform
        };
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
