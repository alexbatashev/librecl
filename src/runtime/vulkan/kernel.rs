use crate::common::cl_types::*;
use crate::common::kernel::Kernel as CommonKernel;

pub struct Kernel {}

impl CommonKernel for Kernel {
    fn set_data_arg(&self, index: usize, bytes: &[u8]) {
        unimplemented!();
    }
    fn set_buffer_arg(&self, index: usize) {
        unimplemented!();
    }
}
