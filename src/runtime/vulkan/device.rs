use vulkano::device::physical::{PhysicalDevice, QueueFamily};

pub struct Device {
    physical_device: Box<PhysicalDevice<'static>>,
}

impl Device {
    pub fn new(physical_device: PhysicalDevice<'static>, queue_family: QueueFamily) -> Device {
        return Device {
            physical_device: Box::new(physical_device),
        };
    }
}

// impl<'a> framework::Device for Device<'a> {}
