use vulkano::device::physical::{PhysicalDevice, QueueFamily};

pub struct Device<'a> {
    physical_device: PhysicalDevice<'a>,
}

impl<'a> Device<'a> {
    pub fn new(physical_device: PhysicalDevice<'a>, queue_family: QueueFamily) -> Device<'a> {
        return Device {
            physical_device: physical_device,
        };
    }
}

impl<'a> framework::Device for Device<'a> {}
