use super::ComputeCommandEncoder;

pub struct CommandBuffer {
}

#[allow(dead_code)]
impl CommandBuffer {
    pub fn new_compute_command_encoder(&self) -> ComputeCommandEncoder {
      unimplemented!()
    }

    pub fn commit(&self) {
      unimplemented!()
    }

    pub fn wait_until_completed(&self) {
      unimplemented!()
    }
}

#[derive(Clone)]
pub struct CommandQueue {
}

#[allow(dead_code)]
impl CommandQueue {
    pub fn new_command_buffer(&self) -> CommandBuffer {
      unimplemented!()
    }
}
