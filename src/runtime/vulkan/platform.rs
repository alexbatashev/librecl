use crate::common::device::ClDevice;
use crate::common::context::ClContext;
use crate::common::platform::ClPlatform;
use crate::vulkan::device::Device;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Arc;
use std::rc::Rc;
use vulkano::{
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device as VkDevice, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
    },
    instance::{Instance, InstanceCreateInfo},
};

static mut VK_INSTANCE: Lazy<Arc<Instance>> = Lazy::new(|| {
    return Instance::new(InstanceCreateInfo {
        ..Default::default()
    })
    .unwrap();
});

pub struct Platform {
    devices: Vec<Rc<ClDevice>>,
    instance: Arc<Instance>,
    platform_name: String,
}

impl Platform {
    pub fn new(
        vendor_name: &str,
        devices: Vec<Rc<ClDevice>>,
        instance: Arc<Instance>,
    ) -> Platform {
        let platform_name = std::format!("LibreCL {} Vulkan Platform", vendor_name);
        return Platform {
            devices,
            instance,
            platform_name,
        };
    }
    pub fn create_platforms(platforms: &mut Vec<Arc<ClPlatform>>) {
        // TODO this should be safer
        let extensions = DeviceExtensions {
            khr_storage_buffer_storage_class: true,
            ..DeviceExtensions::none()
        };

        let devices = PhysicalDevice::enumerate(unsafe { &VK_INSTANCE })
            .filter(|&p| p.supported_extensions().is_superset_of(&extensions))
            .filter_map(|p| {
                p.queue_families()
                    .find(|&q| q.supports_compute())
                    .map(|q| (p, q))
            });

        let mut platform_to_device: HashMap<u32, Vec<Rc<ClDevice>>> = HashMap::new();

        for (device, queue_index) in devices {
            let cl_device = Rc::new(Device::new(device, queue_index).into());

            let id = device.properties().vendor_id;

            if !platform_to_device.contains_key(&id) {
                platform_to_device.insert(id, vec![]);
            }

            platform_to_device
                .entry(id)
                .and_modify(|e| e.push(cl_device));
        }

        for (vendor, devices) in platform_to_device {
            let vendor_name = match vendor {
                4318 => "NVIDIA",
                5045 => "ARM",
                4098 => "AMD",
                32902 => "Intel",
                4203 => "Apple",
                20803 => "Qualcomm",
                _ => "Unknown Vendor",
            };
            let cl_platform = Arc::new(
                Platform::new(vendor_name, devices, unsafe { VK_INSTANCE.clone() }).into(),
            );
            platforms.push(cl_platform);
        }
    }
}

impl crate::common::platform::Platform for Platform {
    fn get_platform_name(&self) -> &str {
        return self.platform_name.as_str();
    }

    fn get_devices(&self) -> &Vec<Rc<ClDevice>> {
        return &self.devices;
    }

    fn add_device(&mut self, device: Rc<ClDevice>) {
        unimplemented!();
    }

    fn create_context(&self, devices: &Vec<Rc<ClDevice>>) -> Rc<ClContext> {
        unimplemented!();
    }
}
