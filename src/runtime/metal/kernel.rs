use crate::common::cl_types::*;
use crate::common::kernel::Kernel as CommonKernel;
use librecl_compiler::{KernelArgInfo, KernelArgType};

pub struct Kernel {}

impl CommonKernel for Kernel {
    fn set_data_arg(&mut self, index: usize, bytes: &[u8]) {
        unimplemented!();
    }
    fn set_buffer_arg(&mut self, index: usize, buffer: cl_mem) {
        unimplemented!();
    }
    fn get_arg_info(&self) -> &[KernelArgInfo] {
        unimplemented!();
    }
}
