use crate::api::cl_types::*;
use crate::interface::{
    ContextKind, DeviceImpl, DeviceKind, DeviceLimits, PlatformKind, QueueKind, VectorCaps,
};
use crate::sync::{self, *};
use cpmetal::Device as MTLDevice;
use librecl_compiler::{CompileResult, Compiler};
use ocl_type_wrapper::ClObjImpl;
use ocl_type_wrapper::DeviceLimitsInterface;
use std::sync::Arc;

use super::InOrderQueue;

const COMMON_BUILTINS: &str = include_str!("../builtin/common.mlir");

#[derive(ClObjImpl, DeviceLimitsInterface)]
pub struct Device {
    platform: WeakPtr<PlatformKind>,
    device: MTLDevice,
    compiler: Arc<Compiler>,
    builtin_libraries: Vec<Arc<CompileResult>>,
    device_limits: DeviceLimits,
    #[cl_handle]
    handle: UnsafeHandle<cl_device_id>,
}

impl Device {
    pub fn new(platform: &SharedPtr<PlatformKind>, device: MTLDevice) -> SharedPtr<DeviceKind> {
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

        let device_limits = DeviceLimits {
            max_compute_units: 1,
            max_work_item_dimensions: 3,
            max_work_item_sizes: [0, 0, 0],
            max_work_group_size: 0,
            preferred_vector_caps: vec_limits.clone(),
            native_vector_caps: vec_limits,
            max_mem_alloc_size: 0,
            preferred_work_group_size_multiple: 32,
        };

        let compiler = Compiler::new();
        let mut builtin_libraries = vec![];

        if compiler.is_available() {
            println!("{}", COMMON_BUILTINS);
            let split_opts = ["--targets=metal-ios,metal-macos".to_owned()];
            let options = ocl_args::parse_options(&split_opts).unwrap();
            builtin_libraries.push(compiler.compile_mlir(COMMON_BUILTINS, &options));
        }

        let device = Device {
            platform: SharedPtr::downgrade(platform),
            device,
            compiler,
            builtin_libraries,
            device_limits,
            handle: UnsafeHandle::null(),
        }
        .into();

        let raw_device = _cl_device_id::wrap(device);
        return DeviceKind::try_from_cl(raw_device).unwrap();
    }

    pub fn get_builtin_libs(&self) -> &[Arc<CompileResult>] {
        &self.builtin_libraries
    }

    pub fn get_native_device(&self) -> &MTLDevice {
        return &self.device;
    }

    pub fn get_compiler(&self) -> Arc<Compiler> {
        self.compiler.clone()
    }
}

impl DeviceImpl for Device {
    fn get_device_type(&self) -> cl_device_type {
        return cl_device_type::GPU;
    }
    fn get_device_name(&self) -> String {
        self.device.name().clone()
    }
    fn is_available(&self) -> bool {
        // TODO some Intel-based Macs support hybrid graphics and eGPUs.
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
        return InOrderQueue::new(
            SharedPtr::downgrade(&context),
            SharedPtr::downgrade(&device),
        )
        .into();
    }

    fn get_vendor_name(&self) -> String {
        unimplemented!()
    }

    fn get_vendor_id(&self) -> cl_uint {
        unimplemented!()
    }

    fn is_compiler_available(&self) -> bool {
        unimplemented!()
    }

    fn get_native_driver_version(&self) -> String {
        unimplemented!()
    }

    fn get_device_profile(&self) -> String {
        unimplemented!()
    }

    fn get_device_version_info(&self) -> String {
        unimplemented!()
    }

    fn get_extension_names(&self) -> &[&str] {
        unimplemented!()
    }

    fn get_extension_versions(&self) -> &[cl_version] {
        unimplemented!()
    }
}

unsafe impl Sync for Device {}
