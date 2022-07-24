use crate::common::cl_types::cl_context;
use crate::common::cl_types::cl_device_id;
use crate::common::cl_types::cl_context_callback;
use crate::common::context::ClContext;
use crate::common::device::ClDevice;
use crate::common::platform::ClPlatform;
use crate::common::platform::Platform as CommonPlatform;
use crate::vulkan::device::Device;
use crate::vulkan::context::Context;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use vulkano::{
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType, QueueFamily},
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
    pub fn new(vendor_name: &str, instance: Arc<Instance>) -> Platform {
        let platform_name = std::format!("LibreCL {} Vulkan Platform", vendor_name);
        return Platform {
            devices: vec![],
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

        let mut platform_to_device: HashMap<u32, Vec<(PhysicalDevice, QueueFamily)>> =
            HashMap::new();

        for (device, queue_index) in devices {
            let id = device.properties().vendor_id;

            if !platform_to_device.contains_key(&id) {
                platform_to_device.insert(id, vec![]);
            }

            platform_to_device
                .entry(id)
                .and_modify(|e| e.push((device, queue_index)));
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
            let cl_platform: Arc<ClPlatform> =
                Arc::new(Platform::new(vendor_name, unsafe { VK_INSTANCE.clone() }).into());

            for device_parts in devices {
                let (device, queue_index) = device_parts;
                let cl_device =
                    Rc::new(Device::new(Arc::downgrade(&cl_platform), device, queue_index).into());
                let ptr: *const ClDevice = Rc::as_ptr(&cl_device);
                unsafe {
                    (Arc::as_ptr(&cl_platform) as *mut ClPlatform)
                        .as_mut()
                        .unwrap()
                        .add_device(cl_device);
                }
            }

            platforms.push(cl_platform);
        }
    }
}

impl CommonPlatform for Platform {
    fn get_platform_name(&self) -> &str {
        return self.platform_name.as_str();
    }

    fn get_devices(&self) -> &Vec<Rc<ClDevice>> {
        return &self.devices;
    }

    fn add_device(&mut self, device: Rc<ClDevice>) {
        self.devices.push(device);
    }

    fn create_context(&self, devices: &[cl_device_id], callback: cl_context_callback, user_data: *mut libc::c_void) -> cl_context {
        return Context::new(devices, callback, user_data);
    }
}
