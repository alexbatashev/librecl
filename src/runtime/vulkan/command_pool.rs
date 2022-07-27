use crate::common::command_pool::NativeCommandPool;
use vulkano::command_buffer::pool::UnsafeCommandPool;

pub struct VkCommandPool {
    pool: CommandPool,
}

impl NativeCommandPool for VkCommandPool {
    type CommandBuffer = VkCommandBuffer;

    fn create_command_buffers(&self, num_buffers: usize) -> Vec<Self::CommandBuffer> {}
}
