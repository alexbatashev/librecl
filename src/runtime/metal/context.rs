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
    fn create_program_with_source(&self, source: String) -> cl_program {
        unimplemented!();
    }
}
