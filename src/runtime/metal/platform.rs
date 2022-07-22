use crate::common::device::ClDevice;
use crate::common::platform::ClPlatform;

use std::sync::Arc;

use metal_api::Device as MTLDevice;

use crate::metal::Device;

pub struct Platform {
    devices: Vec<Arc<ClDevice>>,
    name: String,
}

impl Platform {
    pub fn new(vendor_name: &str) -> Platform {
        let platform_name = std::format!("LibreCL {} Vulkan Platform", vendor_name);
        return Platform {
            devices: vec![],
            name: platform_name,
        };
    }

    fn add_device(&mut self, device: Arc<ClDevice>) {
        self.devices.push(device);
    }

    pub fn create_platforms(platforms: &mut Vec<Arc<ClPlatform>>) {
        let mut platform = Arc::new(Platform::new("Apple").into());

        let all_devices = MTLDevice::all();

        for d in all_devices {
            // add_device(Arc::new(Device::new(platform, d).into()));
        }

        platforms.push(platform);
    }
}

impl crate::common::platform::Platform for Platform {
    fn get_platform_name(&self) -> &str {
        return self.name.as_str();
    }
}
