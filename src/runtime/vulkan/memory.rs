use crate::common::cl_types::*;
use crate::common::context::{ClContext, Context as CommonContext};
use crate::common::device::ClDevice;
use crate::common::memory::MemObject;
use std::sync::Arc;
use vulkano::VulkanObject;
use vulkano::{
    buffer::{vma::VmaBuffer, BufferUsage, CpuAccessibleBuffer, ImmutableBuffer},
    command_buffer::{CommandBufferExecFuture, PrimaryAutoCommandBuffer},
    device::DeviceOwned,
    sync::NowFuture,
};

#[derive(Clone)]
pub struct SingleDeviceBuffer {
    context: cl_context,
    size: usize,
    buffer: Arc<VmaBuffer>,
}

pub struct SingleDeviceImplicitBuffer {
    _context: cl_context,
    _size: usize,
    buffer: Arc<ImmutableBuffer<[u8]>>,
    future: Arc<CommandBufferExecFuture<NowFuture, PrimaryAutoCommandBuffer>>,
}

impl SingleDeviceBuffer {
    pub fn new(
        allocator: Arc<vk_mem::Allocator>,
        context: cl_context,
        size: usize,
    ) -> SingleDeviceBuffer {
        let ctx_safe = unsafe { context.as_ref() }.unwrap();
        let device = match unsafe { ctx_safe.get_associated_devices()[0].as_ref() }.unwrap() {
            ClDevice::Vulkan(device) => device.get_logical_device(),
            _ => panic!("unexpected enum value"),
        };
        let buffer = SingleDeviceBuffer {
            context,
            size,
            buffer: VmaBuffer::allocate(device, allocator, BufferUsage::storage_buffer(), size),
        };
        buffer.buffer.device().internal_object();
        return buffer;
    }

    // TODO support offset and size
    pub fn write(&self, data: *const libc::c_void) {
        let context = match unsafe { self.context.as_ref() }.unwrap() {
            ClContext::Vulkan(context) => context,
            _ => panic!(),
        };
        // TODO errors

        let dst = unsafe { self.buffer.map(context.get_allocator()) };

        unsafe {
            libc::memcpy(dst as *mut libc::c_void, data, self.size);
        }
    }

    pub fn read(&self, data: *mut libc::c_void) {
        let context = match unsafe { self.context.as_ref() }.unwrap() {
            ClContext::Vulkan(context) => context,
            _ => panic!(),
        };
        // TODO errors

        let src = unsafe { self.buffer.map(context.get_allocator()) };

        unsafe {
            libc::memcpy(data, src as *const libc::c_void, self.size);
        }
    }

    pub fn get_buffer(&self) -> Arc<VmaBuffer> {
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
