use ash::vk::{Buffer, BufferCreateInfo, BufferUsageFlags};
use std::sync::Arc;
use vk_mem;
use vulkano::{device::Device, buffer::{BufferUsage, sys::{UnsafeBuffer, BufferState}}, DeviceSize};
use parking_lot::Mutex;

trait CreateFromParts {
    fn create(handle: Buffer, device: Arc<Device>, size: DeviceSize, usage: BufferUsage) -> Arc<UnsafeBuffer>;
}

impl CreateFromParts for UnsafeBuffer {
    fn create(handle: Buffer, device: Arc<Device>, size: DeviceSize, usage: BufferUsage) -> Arc<UnsafeBuffer> {
        return Arc::new(UnsafeBuffer{
            handle,
            device,
            size,
            usage,
            state: Mutex::new(BufferState::new(size)),
        });
    }
}

pub struct VmaBuffer {
    allocation_info: vk_mem::AllocationInfo,
    allocation: vk_mem::Allocation,
    buffer: Arc<UnsafeBuffer>,
}

impl VmaBuffer {
    pub fn allocate(
        device: Arc<Device>,
        allocator: Arc<vk_mem::Allocator>,
        size: usize,
    ) -> Arc<VmaBuffer> {
        let allocation_create_info = vk_mem::AllocationCreateInfo::new().usage(vk_mem::MemoryUsage::CpuToGpu);
        let buffer_create_info = BufferCreateInfo::builder()
            .size(size as u64)
            .usage(BufferUsageFlags::STORAGE_BUFFER)
            .build();
        // TODO error handling
        let (buffer, allocation, allocation_info) =
            unsafe { allocator.create_buffer(&buffer_create_info, &allocation_create_info) }.unwrap();

        return Arc::new(VmaBuffer{
            allocation_info,
            allocation,
            buffer,
            device: device.clone()
        });
    }
}
