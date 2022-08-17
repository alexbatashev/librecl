use crate::ComputeCommandEncoder;
use metal::CommandBuffer as MTLCommandBuffer;
use metal::CommandQueue as MTLQueue;
use std::sync::{Arc, Mutex, Weak};

pub struct CommandBuffer {
    pub(crate) _queue: Weak<Mutex<MTLQueue>>,
    pub(crate) buffer: Arc<Mutex<MTLCommandBuffer>>,
}

impl CommandBuffer {
    pub fn new_compute_command_encoder(&self) -> ComputeCommandEncoder {
        let locked_buffer = self.buffer.lock().unwrap();
        let encoder = locked_buffer.new_compute_command_encoder();
        ComputeCommandEncoder {
            encoder: Arc::new(Mutex::new(encoder.to_owned())),
        }
    }

    pub fn commit(&self) {
        let locked_buffer = self.buffer.lock().unwrap();
        locked_buffer.commit();
    }

    pub fn wait_until_completed(&self) {
        let locked_buffer = self.buffer.lock().unwrap();
        locked_buffer.wait_until_completed();
    }
}

#[derive(Clone)]
pub struct CommandQueue {
    pub(crate) queue: Arc<Mutex<MTLQueue>>,
}

impl CommandQueue {
    pub fn new_command_buffer(&self) -> CommandBuffer {
        let queue = Arc::downgrade(&self.queue);
        let owned_queue = self.queue.lock().unwrap();
        let buffer = owned_queue.new_command_buffer();

        CommandBuffer {
            _queue: queue,
            buffer: Arc::new(Mutex::new(buffer.to_owned())),
        }
    }
}

unsafe impl Sync for CommandQueue {}
unsafe impl Send for CommandQueue {}
