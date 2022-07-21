use crate::device::Device;
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

pub struct Platform {
    devices: Vec<Box<*mut dyn framework::Device>>,
    instance: Arc<Instance>,
}

impl Platform {
    pub fn new(devices: Vec<Box<*mut dyn framework::Device>>, instance: Arc<Instance>) -> Platform {
        return Platform {
            devices: devices,
            instance: instance,
        };
    }
    pub fn create_platforms() -> Vec<Box<*mut Platform>> {
        // TODO this should be safer
        /*let instance = Instance::new(InstanceCreateInfo {
            ..Default::default()
        })
        .unwrap();*/

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

        let mut platform_to_device: HashMap<u32, Vec<Box<*mut dyn framework::Device>>> =
            HashMap::new();

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

        for (_, devices) in platform_to_device {
            let cl_platform = Box::leak(Box::new(Platform::new(devices, unsafe {
                VK_INSTANCE.clone()
            }))) as *mut Platform;
            platforms.push(Box::new(cl_platform));
        }

        return platforms;
    }
}

// impl framework::Platform for Platform {}

#[no_mangle]
pub extern "C" fn clGetPlatformIDs(
    num_entries: cl_uint,
    platforms_raw: *mut *mut c_void,
    num_platforms_raw: *mut cl_uint,
) -> cl_int {
    let num_platforms = unsafe { num_platforms_raw.as_ref() };
    let platforms = unsafe { platforms_raw.as_ref() };

    lcl_contract!(
        num_entries != 0 || (!num_platforms.is_none() && *num_platforms.unwrap() != 0u32),
        "num_entries and num_platforms can not be 0 at the same time",
        1
    );

    lcl_contract!(
        platforms.is_none() && num_platforms.is_none(),
        "num_platforms and platforms can not be NULL at the same time",
        1
    );

    if !platforms.is_none() {
        let platforms_array = unsafe {
            std::slice::from_raw_parts_mut(
                platforms_raw as *mut *mut Platform,
                num_entries as usize,
            )
        };
        for i in 0..num_entries {
            platforms_array[i as usize] = unsafe { *GLOBAL_PLATFORMS[i as usize] };
        }
    }

    if !num_platforms.is_none() {
        unsafe {
            *num_platforms_raw = GLOBAL_PLATFORMS.len() as u32;
        };
    }

    return CL_SUCCESS;
}
