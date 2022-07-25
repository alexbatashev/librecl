use crate::common::cl_types::*;
use crate::common::queue::Queue as CommonQueue;

pub struct InOrderQueue {
    context: cl_context,
    device: cl_device_id,
}

impl InOrderQueue {
    pub fn new(context: cl_context, device: cl_device_id) -> cl_command_queue {
        return Box::leak(Box::new(InOrderQueue { context, device }.into()));
    }
}

impl CommonQueue for InOrderQueue {}
