use metal_api::Device as MTLDevice;

use crate::common::{self, cl_types::cl_device_type, platform::ClPlatform};
use std::sync::{Arc, Weak};

pub struct Device {
    platform: Weak<ClPlatform>,
    device: MTLDevice,
}

impl Device {
    pub fn new(platform: &Arc<ClPlatform>, device: MTLDevice) -> Device {
        return Device {
            platform: Arc::downgrade(platform),
            device: device,
        };
    }
}

impl common::device::Device for Device {
    fn get_device_type(&self) -> cl_device_type {
        return cl_device_type::GPU;
    }
    fn get_device_name(&self) -> String {
        unimplemented!();
    }
}
