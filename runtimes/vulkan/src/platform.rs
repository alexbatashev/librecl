use crate::device::Device;
use api_generator::cl_get_platform_ids;
use framework::*;
use libc::c_void;
use once_cell::sync::Lazy;
use std::alloc::Layout;
use std::collections::HashMap;
use std::sync::Arc;
use vulkano::{
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device as VkDevice, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
    },
    instance::{Instance, InstanceCreateInfo},
};

macro_rules! box_array {
    ($val:expr ; $len:expr) => {{
        // Use a generic function so that the pointer cast remains type-safe
        fn vec_to_boxed_array<T>(vec: Vec<T>) -> Box<[T; $len]> {
            let boxed_slice = vec.into_boxed_slice();

            let ptr = ::std::boxed::Box::into_raw(boxed_slice) as *mut [T; $len];

            unsafe { Box::from_raw(ptr) }
        }

        vec_to_boxed_array(vec![$val; $len])
    }};
}

static mut GLOBAL_PLATFORMS: Lazy<Vec<Box<*mut Platform>>> =
    Lazy::new(|| Platform::create_platforms());
static mut VK_INSTANCE: Lazy<Arc<Instance>> = Lazy::new(|| {
    return Instance::new(InstanceCreateInfo {
        ..Default::default()
    })
    .unwrap();
});

pub struct Platform<'a> {
    devices: Vec<Box<*mut Device<'a>>>,
    instance: Arc<Instance>,
    name: &str,
}

impl<'a> Platform<'a> {
    pub fn new(
        vendor_name: &str,
        devices: Vec<Box<*mut Device<'a>>>,
        instance: Arc<Instance>,
    ) -> Platform<'a> {
        let platform_name = std::format!("LibreCL {} Vulkan Platform".vendor_name);
        return Platform {
            devices: devices,
            instance: instance,
            name: &platform_name.clone(),
        };
    }
    pub fn create_platforms() -> Vec<Box<*mut Platform<'a>>> {
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

        let mut platform_to_device: HashMap<u32, Vec<Box<*mut Device>>> = HashMap::new();

        for (device, queue_index) in devices {
            let cl_device = Box::leak(Box::new(Device::new(device, queue_index))) as *mut Device;

            let id = device.properties().vendor_id;

            if !platform_to_device.contains_key(&id) {
                platform_to_device.insert(id, vec![]);
            }

            platform_to_device
                .entry(id)
                .and_modify(|e| e.push(Box::new(cl_device)));
        }

        let mut platforms = vec![];

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
            let cl_platform = Box::leak(Box::new(Platform::new(vendor_name, devices, unsafe {
                VK_INSTANCE.clone()
            }))) as *mut Platform;
            platforms.push(Box::new(cl_platform));
        }

        return platforms;
    }
}

impl<'a> framework::Platform for Platform<'a> {
    fn get_platform_name(&self) -> &str {
        return platform_name;
    }
}

#[cl_get_platform_ids]
fn get_platforms() -> Vec<Box<*mut Platform<'static>>> {
    return unsafe { GLOBAL_PLATFORMS.clone() };
}
