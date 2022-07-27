use crate::common::cl_types::*;
use crate::common::context::{ClContext, Context};
use crate::common::device::ClDevice;
use crate::common::memory::MemObject;
use metal_api::MTLResourceOptions;
use metal_api::{Buffer, ComputeCommandEncoderRef};
use std::sync::Arc;

#[derive(Clone)]
pub struct SingleDeviceBuffer {
    _context: cl_context,
    size: usize,
    buffer: Arc<Buffer>,
}

impl SingleDeviceBuffer {
    pub fn new(context: cl_context, size: usize) -> cl_mem {
        let ctx_safe = match unsafe { context.as_ref() }.unwrap() {
            ClContext::Metal(ctx) => ctx,
            _ => panic!(),
        };

        let device = match unsafe { ctx_safe.get_associated_devices()[0].as_ref() }.unwrap() {
            ClDevice::Metal(dev) => dev,
            _ => panic!(),
        };

        let options = MTLResourceOptions::empty();
        let buffer = Arc::new(device.get_native_device().new_buffer(size as u64, options));

        return Box::into_raw(Box::new(
            SingleDeviceBuffer {
                _context: context,
                size,
                buffer,
            }
            .into(),
        ));
    }

    pub fn encode_argument(&self, command_encoder: &ComputeCommandEncoderRef, idx: usize) {
        command_encoder.set_buffer(idx as u64, Some(self.buffer.as_ref()), 0);
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

impl MemObject for SingleDeviceBuffer {}
