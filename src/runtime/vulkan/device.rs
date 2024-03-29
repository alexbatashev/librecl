use super::queue::InOrderQueue;
use crate::api::cl_types::ClObjectImpl;
use crate::api::cl_types::*;
use crate::interface::ContextKind;
use crate::interface::DeviceKind;
use crate::interface::DeviceLimits;
use crate::interface::QueueKind;
use crate::interface::VectorCaps;
use crate::interface::{DeviceImpl, PlatformKind};
use crate::sync::{self, SharedPtr, UnsafeHandle, WeakPtr};
use librecl_compiler::{CompileResult, Compiler};
use ocl_type_wrapper::ClObjImpl;
use ocl_type_wrapper::DeviceLimitsInterface;
use std::sync::Arc;
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType, QueueFamily};
use vulkano::device::Queue;
use vulkano::device::{Device as VkDevice, DeviceCreateInfo, Features, QueueCreateInfo};

const COMMON_BUILTINS: &str = include_str!("../builtin/common.mlir");

#[derive(ClObjImpl, Clone, DeviceLimitsInterface)]
pub struct Device {
    vendor_name: String,
    physical_device: PhysicalDevice<'static>,
    logical_device: Arc<VkDevice>,
    device_type: cl_device_type,
    platform: WeakPtr<PlatformKind>,
    queue: Arc<Queue>,
    compiler: Arc<Compiler>,
    builtin_libraries: Vec<Arc<CompileResult>>,
    extension_names: Vec<&'static str>,
    extension_versions: Vec<cl_version>,
    device_limits: DeviceLimits,
    #[cl_handle]
    handle: UnsafeHandle<cl_device_id>,
}

impl Device {
    pub fn new(
        platform: WeakPtr<PlatformKind>,
        physical_device: PhysicalDevice<'static>,
        queue_family: QueueFamily,
        vendor_name: &str,
    ) -> SharedPtr<DeviceKind> {
        // TODO figure out if we need queues at all
        let features = Features {
            shader_int64: true,
            ..Features::none()
        };
        let (device, mut queues) = VkDevice::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
                enabled_features: features,
                ..Default::default()
            },
        )
        .expect("could not create a device");

        let device_type = match physical_device.properties().device_type {
            PhysicalDeviceType::Cpu => cl_device_type::CPU,
            PhysicalDeviceType::IntegratedGpu => cl_device_type::GPU,
            PhysicalDeviceType::DiscreteGpu => cl_device_type::GPU,
            PhysicalDeviceType::VirtualGpu => cl_device_type::GPU,
            PhysicalDeviceType::Other => cl_device_type::ACC,
        };

        let extension_names = vec![
            "cl_khr_byte_addressable_store",
            "cl_khr_global_int32_base_atomics",
            "cl_khr_global_int32_extended_atomics",
            "cl_khr_local_int32_base_atomics",
            "cl_khr_local_int32_extended_atomics",
            "cl_khr_il_program",
        ];
        let extension_versions = vec![
            make_version(1, 0, 0),
            make_version(1, 0, 0),
            make_version(1, 0, 0),
            make_version(1, 0, 0),
            make_version(1, 0, 0),
            make_version(1, 0, 0),
        ];

        // TODO figure out real limits
        let vec_limits = VectorCaps {
            vector_width_char: 1,
            vector_width_short: 1,
            vector_width_int: 1,
            vector_width_long: 1,
            vector_width_float: 1,
            vector_width_double: 1,
            vector_width_half: 1,
        };

        let wg_sizes = [
            physical_device.properties().max_compute_work_group_size[0] as cl_size_t,
            physical_device.properties().max_compute_work_group_size[1] as cl_size_t,
            physical_device.properties().max_compute_work_group_size[2] as cl_size_t,
        ];

        let device_limits = DeviceLimits {
            max_compute_units: 1, // TODO this must be obtained from some table
            max_work_item_dimensions: 3,
            max_work_item_sizes: wg_sizes,
            max_work_group_size: physical_device
                .properties()
                .max_compute_work_group_invocations as cl_size_t,
            preferred_vector_caps: vec_limits.clone(),
            native_vector_caps: vec_limits,
            max_mem_alloc_size: physical_device.properties().max_buffer_size.unwrap_or(0),
            preferred_work_group_size_multiple: 32,
        };

        let compiler = Compiler::new();
        let mut builtin_libraries = vec![];

        if compiler.is_available() {
            println!("{}", COMMON_BUILTINS);
            let split_opts = ["--targets=vulkan-spirv".to_owned()];
            let options = ocl_args::parse_options(&split_opts).unwrap();
            builtin_libraries.push(compiler.compile_mlir(COMMON_BUILTINS, &options));
        }

        let device = Device {
            vendor_name: vendor_name.to_owned(),
            physical_device,
            logical_device: device.clone(),
            device_type,
            platform,
            queue: queues.next().unwrap(),
            compiler,
            builtin_libraries,
            extension_names,
            extension_versions,
            device_limits,
            handle: UnsafeHandle::null(),
        }
        .into();
        // This is intentional
        let raw_device = _cl_device_id::wrap(device);
        return DeviceKind::try_from_cl(raw_device).unwrap();
    }

    pub fn get_builtin_libs(&self) -> &[Arc<CompileResult>] {
        &self.builtin_libraries
    }

    pub fn get_logical_device(&self) -> Arc<VkDevice> {
        return self.logical_device.clone();
    }

    pub fn get_physical_device(&self) -> PhysicalDevice {
        return self.physical_device.clone();
    }

    pub fn get_queue(&self) -> Arc<Queue> {
        return self.queue.clone();
    }

    pub fn get_compiler(&self) -> Arc<Compiler> {
        self.compiler.clone()
    }
}

impl DeviceImpl for Device {
    fn get_device_type(&self) -> cl_device_type {
        return self.device_type;
    }

    fn get_device_name(&self) -> String {
        return self.physical_device.properties().device_name.clone();
    }

    fn is_available(&self) -> bool {
        // TODO modern laptops allow external GPUs to be connected
        return true;
    }
    fn get_platform(&self) -> WeakPtr<PlatformKind> {
        return self.platform.clone();
    }

    fn create_queue(
        &self,
        context: SharedPtr<ContextKind>,
        device: SharedPtr<DeviceKind>,
    ) -> QueueKind {
        // TODO support OoO queues;
        return InOrderQueue::new(
            SharedPtr::downgrade(&context),
            SharedPtr::downgrade(&device),
        )
        .into();
    }

    fn get_vendor_name(&self) -> String {
        self.vendor_name.clone()
    }

    fn get_vendor_id(&self) -> cl_uint {
        self.physical_device.properties().vendor_id
    }

    fn is_compiler_available(&self) -> bool {
        self.compiler.is_available()
    }

    fn get_native_driver_version(&self) -> String {
        self.physical_device
            .properties()
            .driver_info
            .as_ref()
            .unwrap_or(&"unknown driver".to_owned())
            .clone()
    }

    fn get_device_profile(&self) -> String {
        if self.is_compiler_available() {
            "FULL_PROFILE".to_owned()
        } else {
            "EMBEDDED_PROFILE".to_owned()
        }
    }

    fn get_device_version_info(&self) -> String {
        format!(
            "Vulkan {:?} - {:?}",
            self.physical_device.api_version(),
            self.physical_device.properties().driver_info
        )
    }

    fn get_extension_names(&self) -> &[&str] {
        &self.extension_names
    }

    fn get_extension_versions(&self) -> &[cl_version] {
        &self.extension_versions
    }
}
