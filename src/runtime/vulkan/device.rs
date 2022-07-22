use crate::common::cl_types::cl_device_type;
use std::sync::Arc;
use vulkano::device::physical::{PhysicalDevice, QueueFamily};
use vulkano::device::{Device as VkDevice, DeviceCreateInfo, QueueCreateInfo};

pub struct Device {
    physical_device: PhysicalDevice<'static>,
    logical_device: Arc<VkDevice>,
    device_type: cl_device_type,
}

impl Device {
    pub fn new(physical_device: PhysicalDevice<'static>, queue_family: QueueFamily) -> Device {
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
        };
    }
}

impl crate::common::device::Device for Device {
    fn get_device_type(&self) -> cl_device_type {
        return self.device_type;
    }

    fn get_device_name(&self) -> String {
        return self.physical_device.properties().device_name.clone();
    }
}
