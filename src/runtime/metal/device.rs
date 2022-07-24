use metal_api::Device as MTLDevice;

use crate::common::{self, platform::ClPlatform};
use crate::common::cl_types::*;
use std::sync::{Arc, Weak};

pub struct Device {
    platform: Weak<ClPlatform>,
    device: MTLDevice,
}

impl Device {
    pub fn new(platform: &Arc<ClPlatform>, device: MTLDevice) -> Device {
        return Device {
            platform: Arc::downgrade(platform),
            device,
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
    fn is_available(&self) -> bool {
        // TODO some Intel-based Macs support hybrid graphics and eGPUs.
        return true;
    }
    fn get_platform(&self) -> cl_platform_id {
        unimplemented!();
    }
    fn create_queue(&self, context: cl_context, device: cl_device_id) -> cl_command_queue {
        unimplemented!();
    }
}
