use metal_api::Device as MTLDevice;
use ocl_type_wrapper::ClObjImpl;

use crate::api::cl_types::*;
use crate::interface::{ContextKind, DeviceImpl, DeviceKind, PlatformKind, QueueKind};
use crate::sync::{self, *};
use std::sync::Arc;
use std::sync::Mutex;

use super::InOrderQueue;

#[derive(ClObjImpl)]
pub struct Device {
    platform: WeakPtr<PlatformKind>,
    device: Arc<Mutex<UnsafeHandle<MTLDevice>>>,
    handle: UnsafeHandle<cl_device_id>,
}

impl Device {
    pub fn new(platform: &SharedPtr<PlatformKind>, device: MTLDevice) -> SharedPtr<DeviceKind> {
        let device = Device {
            platform: SharedPtr::downgrade(platform),
            device: Arc::new(Mutex::new(UnsafeHandle::new(device))),
            handle: UnsafeHandle::null(),
        }
        .into();

        let raw_device = _cl_device_id::wrap(device);
        return DeviceKind::try_from_cl(raw_device).unwrap();
    }

    pub fn get_native_device(&self) -> &Mutex<UnsafeHandle<MTLDevice>> {
        return &self.device;
    }
}

impl DeviceImpl for Device {
    fn get_device_type(&self) -> cl_device_type {
        return cl_device_type::GPU;
    }
    fn get_device_name(&self) -> String {
        let locked_device = self.device.lock().unwrap();
        return locked_device.value().name().to_owned();
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
}

unsafe impl Sync for Device {}
