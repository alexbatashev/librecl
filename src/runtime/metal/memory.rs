use crate::api::cl_types::*;
use crate::interface::{ContextImpl, ContextKind, DeviceKind, MemImpl, MemKind};
use crate::sync::{self, *};
use metal_api::MTLResourceOptions;
use metal_api::{Buffer, ComputeCommandEncoderRef};
use ocl_type_wrapper::ClObjImpl;
use std::ops::Deref;
use std::sync::{Arc, Mutex};

#[derive(Clone, ClObjImpl)]
pub struct SingleDeviceBuffer {
    _context: WeakPtr<ContextKind>,
    size: usize,
    buffer: Arc<Mutex<UnsafeHandle<Buffer>>>,
    handle: UnsafeHandle<cl_mem>,
}

impl SingleDeviceBuffer {
    pub fn new(context: WeakPtr<ContextKind>, size: usize) -> MemKind {
        let owned_context = context.upgrade().unwrap();
        let ctx_safe = match owned_context.deref() {
            ContextKind::Metal(ctx) => ctx,
            #[allow(unreachable_patterns)]
            _ => panic!(),
        };

        let owned_device = ctx_safe.get_associated_devices()[0].upgrade().unwrap();
        let device = match owned_device.deref() {
            DeviceKind::Metal(dev) => dev,
            #[allow(unreachable_patterns)]
            _ => panic!(),
        };

        let options = MTLResourceOptions::empty();
        let locked_device = device.get_native_device().lock().unwrap();
        let buffer = Arc::new(Mutex::new(UnsafeHandle::new(
            locked_device.value().new_buffer(size as u64, options),
        )));

        SingleDeviceBuffer {
            _context: context,
            size,
            buffer,
            handle: UnsafeHandle::null(),
        }
        .into()
    }

    pub fn encode_argument(&self, command_encoder: &ComputeCommandEncoderRef, idx: usize) {
        let locked_buffer = self.buffer.lock().unwrap();
        command_encoder.set_buffer(idx as u64, Some(locked_buffer.value().as_ref()), 0);
    }

    // TODO support offset and size
    pub fn write(&self, data: *const libc::c_void) {
        // TODO errors

        let buffer_lock = self.buffer.lock().unwrap();

        let dst = buffer_lock.value().contents();

        unsafe {
            libc::memcpy(dst, data, self.size);
        }
    }

    pub fn read(&self, data: *mut libc::c_void) {
        // TODO errors

        let buffer_lock = self.buffer.lock().unwrap();
        let src = buffer_lock.value().contents();

        unsafe {
            libc::memcpy(data, src as *const libc::c_void, self.size);
        }
    }
}

impl MemImpl for SingleDeviceBuffer {}
