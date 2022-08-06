use crate::api::cl_types::*;
use crate::interface::ContextKind;
use crate::interface::DeviceKind;
use crate::interface::PlatformImpl;
use crate::interface::PlatformKind;
use crate::sync::{self, SharedPtr, UnsafeHandle, WeakPtr};
use crate::vulkan::context::Context;
use crate::vulkan::device::Device;
use ocl_type_wrapper::ClObjImpl;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::ops::DerefMut;
use std::sync::Arc;
use vulkano::instance::debug::DebugUtilsMessageSeverity;
use vulkano::instance::debug::DebugUtilsMessageType;
use vulkano::instance::debug::DebugUtilsMessengerCreateInfo;
use vulkano::{
    device::{
        physical::{PhysicalDevice, QueueFamily},
        DeviceExtensions,
    },
    instance::{debug::Message, Instance, InstanceCreateInfo, InstanceExtensions},
};

fn debug_message_handler(msg: &Message) {
    println!("[VULKAN]: {}", msg.description);
}

static mut VK_INSTANCE: Lazy<Arc<Instance>> = Lazy::new(|| {
    // TODO enable layers correctly
    /*
    let layers: Vec<_> = instance::layers_list()
        .unwrap()
        .filter(|l| l.description().contains("VK_LAYER_KHRONOS_validation"))
        .collect();
    */
    let extensions = InstanceExtensions {
        ext_debug_utils: true,
        ..InstanceExtensions::none()
    };
    let instance_create_info = InstanceCreateInfo {
        enabled_layers: [String::from("VK_LAYER_KHRONOS_validation")].to_vec(),
        enabled_extensions: extensions,
        ..Default::default()
    };
    let mut debug_utils_messengers =
        DebugUtilsMessengerCreateInfo::user_callback(Arc::new(debug_message_handler));
    debug_utils_messengers.message_severity = DebugUtilsMessageSeverity::errors_and_warnings();
    debug_utils_messengers.message_type = DebugUtilsMessageType::all();
    return unsafe {
        Instance::with_debug_utils_messengers(instance_create_info, [debug_utils_messengers])
    }
    .unwrap();
});

#[derive(ClObjImpl)]
pub struct Platform {
    devices: Vec<SharedPtr<DeviceKind>>,
    platform_name: String,
    extension_names: Vec<&'static str>,
    extension_versions: Vec<cl_version>,
    #[cl_handle]
    handle: UnsafeHandle<cl_platform_id>,
}

impl Platform {
    pub fn new(vendor_name: &str) -> Platform {
        let platform_name = std::format!("LibreCL {} Vulkan Platform", vendor_name);

        let extension_names = vec!["cl_khr_icd"];
        let extension_versions = vec![make_version(1, 0, 0)];

        return Platform {
            devices: vec![],
            platform_name,
            extension_names,
            extension_versions,
            handle: UnsafeHandle::null(),
        };
    }
    pub fn create_platforms(platforms: &mut Vec<SharedPtr<PlatformKind>>) {
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

            let platform = Platform::new(vendor_name).into();
            let raw_platform = _cl_platform_id::wrap(platform);
            let mut cl_platform = PlatformKind::try_from_cl(raw_platform).unwrap();

            for device_parts in devices {
                let (device, queue_index) = device_parts;
                let cl_device =
                    Device::new(SharedPtr::downgrade(&cl_platform), device, queue_index).into();
                cl_platform.deref_mut().add_device(cl_device);
            }

            platforms.push(cl_platform);
        }
    }
}

impl PlatformImpl for Platform {
    fn get_devices(&self) -> &[SharedPtr<DeviceKind>] {
        return &self.devices.as_slice();
    }

    fn add_device(&mut self, device: SharedPtr<DeviceKind>) {
        self.devices.push(device);
    }

    fn create_context(
        &self,
        devices: &[WeakPtr<DeviceKind>],
        callback: cl_context_callback,
        user_data: *mut libc::c_void,
    ) -> ContextKind {
        return Context::new(devices, callback, user_data);
    }

    fn get_profile(&self) -> &str {
        return "FULL_PROFILE";
    }

    fn get_platform_version_info(&self) -> &str {
        // TODO think how to return platform version from here.
        return "over Vulkan";
    }

    fn get_platform_name(&self) -> &str {
        return self.platform_name.as_str();
    }

    fn get_extension_names(&self) -> &[&str] {
        return &self.extension_names;
    }

    fn get_extension_versions(&self) -> &[cl_version] {
        return &self.extension_versions;
    }

    fn get_host_timer_resolution(&self) -> cl_ulong {
        return 0;
    }
}
