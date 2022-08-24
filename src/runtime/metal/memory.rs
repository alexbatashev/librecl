use crate::api::cl_types::*;
use crate::interface::{ContextImpl, ContextKind, DeviceKind, MemImpl, MemKind};
use crate::sync::{self, *};
use cpmetal::ResourceOptions;
use cpmetal::{Buffer as MTLBuffer, ComputeCommandEncoder};
use lcl_derive::ClObjImpl;
use std::ops::Deref;

#[derive(Clone, ClObjImpl)]
pub struct SingleDeviceBuffer {
    _context: WeakPtr<ContextKind>,
    size: usize,
    buffer: MTLBuffer,
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

        let options = ResourceOptions::empty();
        let buffer = device.get_native_device().new_buffer(size, &options);

        SingleDeviceBuffer {
            _context: context,
            size,
            buffer,
            handle: UnsafeHandle::null(),
        }
        .into()
    }

    pub fn encode_argument(&self, command_encoder: &ComputeCommandEncoder, idx: usize) {
        command_encoder.set_buffer(idx, Some(&self.buffer), 0);
    }

    // TODO support offset and size
    pub fn write(&self, data: *const libc::c_void) {
        // TODO errors

        let dst = self.buffer.contents();

        unsafe {
            libc::memcpy(dst, data, self.size);
        }
    }

    pub fn read(&self, data: *mut libc::c_void) {
        // TODO errors

        let src = self.buffer.contents();

        unsafe {
            libc::memcpy(data, src as *const libc::c_void, self.size);
        }
    }
}

impl MemImpl for SingleDeviceBuffer {}
