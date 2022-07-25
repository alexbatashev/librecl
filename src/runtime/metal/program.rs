use crate::common::cl_types::*;
use crate::common::program::Program as CommonProgram;

pub struct Program {}

impl CommonProgram for Program {
    fn get_context() -> cl_context {
        unimplemented!();
    }
    fn compile_program(&self, devices: &[&ClDevice]) -> bool {
        unimplemented!();
    }
    fn link_programs(&self, devices: &[&ClDevice]) -> bool {
        unimplemented!();
    }

    fn create_kernel(&self, kernel_name: &str) -> cl_kernel {
        unimplemented!();
    }
}
