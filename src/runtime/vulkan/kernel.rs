use librecl_compiler::KernelArgInfo;

use crate::common::cl_types::*;
use crate::common::kernel::Kernel as CommonKernel;

pub struct Kernel {
    name: String,
    args: Vec<KernelArgInfo>,
}

impl Kernel {
    pub fn new(name: String, args: Vec<KernelArgInfo>) -> Kernel {
        return Kernel { name, args };
    }
}

impl CommonKernel for Kernel {
    fn set_data_arg(&self, index: usize, bytes: &[u8]) {
        unimplemented!();
    }
    fn set_buffer_arg(&self, index: usize) {
        unimplemented!();
    }
}
