use crate::common::cl_types::*;
use crate::common::context::Context as CommonContext;

pub struct Context {}

impl CommonContext for Context {
    fn notify_error(&self, message: String) {
        unimplemented!();
    }
    fn has_device(&self, device: cl_device_id) -> bool {
        unimplemented!();
    }
    fn create_program_with_source(&self, context: cl_context, source: String) -> cl_program {
        unimplemented!();
    }
    fn create_buffer(&self, context: cl_context, size: usize, flags: cl_mem_flags) -> cl_mem {
        unimplemented!();
    }
}
