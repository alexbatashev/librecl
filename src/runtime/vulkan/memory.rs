use crate::common::cl_types::*;
use crate::common::context::Context as CommonContext;
use crate::common::device::ClDevice;
use crate::common::memory::MemObject;
use std::sync::Arc;
use vulkano::VulkanObject;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer},
    command_buffer::{CommandBufferExecFuture, PrimaryAutoCommandBuffer},
    device::DeviceOwned,
    sync::NowFuture,
};

#[derive(Clone)]
pub struct SingleDeviceBuffer {
    _context: cl_context,
    size: usize,
    buffer: Arc<CpuAccessibleBuffer<[u8]>>,
}

pub struct SingleDeviceImplicitBuffer {
    _context: cl_context,
    _size: usize,
    buffer: Arc<ImmutableBuffer<[u8]>>,
    future: Arc<CommandBufferExecFuture<NowFuture, PrimaryAutoCommandBuffer>>,
}

impl SingleDeviceBuffer {
    pub fn new(context: cl_context, size: usize) -> SingleDeviceBuffer {
        let ctx_safe = unsafe { context.as_ref() }.unwrap();
        let device = match unsafe { ctx_safe.get_associated_devices()[0].as_ref() }.unwrap() {
            ClDevice::Vulkan(device) => device.get_logical_device(),
            _ => panic!("unexpected enum value"),
        };
        let buffer = SingleDeviceBuffer {
            _context: context,
            size,
            buffer: unsafe {
                CpuAccessibleBuffer::<[u8]>::uninitialized_array(
                    device.clone(),
                    size as u64,
                    BufferUsage::all(),
                    false,
                )
            }
            .unwrap(),
        };
        buffer.buffer.device().internal_object();
        return buffer;
    }

    pub fn write(&self, data: *const libc::c_void) {
        // TODO errors
        let mut lock = self.buffer.write().unwrap();

        let data_slice = unsafe { std::slice::from_raw_parts(data as *const u8, self.size) };

        lock.clone_from_slice(data_slice);
    }

    pub fn read(&self, data: *mut libc::c_void) {
        // TODO errors
        let lock = self.buffer.read().unwrap();

        let data_slice = unsafe { std::slice::from_raw_parts_mut(data as *mut u8, self.size) };

        data_slice.clone_from_slice(lock.as_ref());
    }

    pub fn get_buffer(&self) -> Arc<CpuAccessibleBuffer<[u8]>> {
        return self.buffer.clone();
    }
}

impl MemObject for SingleDeviceBuffer {}

impl SingleDeviceImplicitBuffer {
    pub fn new(context: cl_context, data: Vec<u8>) -> SingleDeviceImplicitBuffer {
        let size = data.len();
        let ctx_safe = unsafe { context.as_ref() }.unwrap();
        let queue = match unsafe { ctx_safe.get_associated_devices()[0].as_ref() }.unwrap() {
            ClDevice::Vulkan(device) => device.get_queue(),
            _ => panic!("unexpected enum value"),
        };
        let (buffer, future) = ImmutableBuffer::from_iter(data, BufferUsage::all(), queue).unwrap();

        return SingleDeviceImplicitBuffer {
            _context: context,
            _size: size,
            buffer,
            future: Arc::new(future),
        };
    }

    pub fn get_future(&self) -> &CommandBufferExecFuture<NowFuture, PrimaryAutoCommandBuffer> {
        return self.future.as_ref();
    }
    pub fn get_buffer(&self) -> Arc<ImmutableBuffer<[u8]>> {
        return self.buffer.clone();
    }
}
