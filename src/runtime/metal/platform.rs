#![feature(get_mut_unchecked)]

use crate::common::device::ClDevice;
use crate::common::platform::ClPlatform;
use crate::common::platform::Platform as CommonPlatform;

use std::sync::Arc;

use metal_api::Device as MTLDevice;

use crate::metal::Device;

pub struct Platform {
    devices: Vec<Arc<ClDevice>>,
    name: String,
}

impl Platform {
    pub fn new() -> Platform {
        let platform_name = std::format!("LibreCL Apple Metal Platform");
        return Platform {
            devices: vec![],
            name: platform_name,
        };
    }

    pub fn create_platforms(platforms: &mut Vec<Arc<ClPlatform>>) {
        let mut platform: Arc<ClPlatform> = Arc::new(Platform::new().into());

        let all_devices = MTLDevice::all();

        for d in all_devices {
            let device: Arc<ClDevice> = Arc::new(Device::new(&platform, d).into());
            unsafe {
                (Arc::as_ptr(&mut platform) as *mut ClPlatform)
                    .as_mut()
                    .unwrap()
                    .add_device(device);
            }
        }

        platforms.push(platform);
    }
}

impl CommonPlatform for Platform {
    fn get_platform_name(&self) -> &str {
        return self.name.as_str();
    }

    fn get_devices(&self) -> &Vec<Arc<ClDevice>> {
        return &self.devices;
    }

    fn add_device(&mut self, device: Arc<ClDevice>) {
        self.devices.push(device);
    }
}
