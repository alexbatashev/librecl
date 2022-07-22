use metal_api::Device as MTLDevice;

use crate::common::{self, platform::ClPlatform};
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

impl common::device::Device for Device {}
