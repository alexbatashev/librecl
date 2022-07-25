use crate::common::cl_types::*;
use crate::common::context::Context as CommonContext;
use crate::common::device::ClDevice;
use crate::common::memory::MemObject;
use vulkano::buffer::{BufferUsage, CpuBufferPool};

pub struct SingleDeviceBuffer {
    context: cl_context,
    buffer: CpuBufferPool<u8>,
}

impl SingleDeviceBuffer {
    pub fn new(context: cl_context) -> SingleDeviceBuffer {
        let ctx_safe = unsafe { context.as_ref() }.unwrap();
        let device = match unsafe { ctx_safe.get_associated_devices()[0].as_ref() }.unwrap() {
            ClDevice::Vulkan(device) => device.get_logical_device(),
            _ => panic!("unexpected enum value"),
        };
        return SingleDeviceBuffer {
            context,
            buffer: CpuBufferPool::new(device, BufferUsage::storage_buffer()),
        };
    }
}

impl MemObject for SingleDeviceBuffer {}
