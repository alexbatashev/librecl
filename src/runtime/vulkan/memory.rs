use crate::api::cl_types::*;
use crate::interface::{ContextImpl, ContextKind, DeviceKind, MemImpl};
use crate::sync::{self, SharedPtr, UnsafeHandle, WeakPtr};
use ocl_type_wrapper::ClObjImpl;
use std::ops::Deref;
use std::sync::Arc;
use vulkano::{
    buffer::{vma::VmaBuffer, BufferUsage, CpuAccessibleBuffer, ImmutableBuffer},
    command_buffer::{CommandBufferExecFuture, PrimaryAutoCommandBuffer},
    sync::NowFuture,
};

#[derive(ClObjImpl)]
pub struct SingleDeviceBuffer {
    context: WeakPtr<ContextKind>,
    size: usize,
    buffer: Arc<VmaBuffer>,
    #[cl_handle]
    handle: UnsafeHandle<cl_mem>,
}

pub struct SingleDeviceImplicitBuffer {
    _context: WeakPtr<ContextKind>,
    _size: usize,
    buffer: Arc<ImmutableBuffer<[u8]>>,
    future: Arc<CommandBufferExecFuture<NowFuture, PrimaryAutoCommandBuffer>>,
}

impl SingleDeviceBuffer {
    pub fn new(
        allocator: Arc<vk_mem::Allocator>,
        context: WeakPtr<ContextKind>,
        size: usize,
    ) -> SingleDeviceBuffer {
        let owned_context = context.upgrade().unwrap();
        let owned_device = owned_context.get_associated_devices()[0].upgrade().unwrap();
        let device = match owned_device.deref() {
            DeviceKind::Vulkan(device) => device.get_logical_device(),
            _ => panic!("unexpected enum value"),
        };
        let buffer = SingleDeviceBuffer {
            context,
            size,
            buffer: VmaBuffer::allocate(device, allocator, BufferUsage::storage_buffer(), size),
            handle: UnsafeHandle::null(),
        };
        return buffer;
    }

    // TODO support offset and size
    pub fn write(&self, data: *const libc::c_void) {
        let owned_context = self.context.upgrade().unwrap();
        let context = match owned_context.deref() {
            ContextKind::Vulkan(context) => context,
            _ => panic!(),
        };
        // TODO errors

        let dst = unsafe { self.buffer.map(context.get_allocator()) };

        unsafe {
            libc::memcpy(dst as *mut libc::c_void, data, self.size);
        }
    }

    pub fn read(&self, data: *mut libc::c_void) {
        let owned_context = self.context.upgrade().unwrap();
        let context = match owned_context.deref() {
            ContextKind::Vulkan(context) => context,
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

impl MemImpl for SingleDeviceBuffer {}

impl SingleDeviceImplicitBuffer {
    pub fn new(context: WeakPtr<ContextKind>, data: Vec<u8>) -> SingleDeviceImplicitBuffer {
        let size = data.len();
        let owned_context = context.upgrade().unwrap();
        let owned_device = owned_context.get_associated_devices()[0].upgrade().unwrap();
        let queue = match owned_device.deref() {
            DeviceKind::Vulkan(device) => device.get_queue(),
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
