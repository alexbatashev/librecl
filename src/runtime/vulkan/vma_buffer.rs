use ash::vk::Buffer;
use std::sync::Arc;
use vk_mem;
use vulkano::device::Device;

pub struct VmaBuffer {
    allocation_info: vk_mem::AllocationInfo,
    buffer: Buffer,
    device: Arc<Device>,
}

impl VmaBuffer {
    pub fn allocate(size: usize) -> Arc<VmaBuffer> {
        unimplemented!();
    }
}
