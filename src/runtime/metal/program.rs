use crate::common::cl_types::*;
use crate::common::context::ClContext;
use crate::common::device::ClDevice;
use crate::common::program::Program as CommonProgram;

pub struct Program {}

impl CommonProgram for Program {
    fn get_context(&self) -> cl_context {
        unimplemented!();
    }
    fn get_safe_context_mut<'a, 'b>(&'a mut self) -> &'b mut ClContext {
        unimplemented!();
    }
    fn compile_program(&mut self, devices: &[&ClDevice]) -> bool {
        unimplemented!();
    }
    fn link_programs(&mut self, devices: &[&ClDevice]) -> bool {
        unimplemented!();
    }

    fn create_kernel(&self, program: cl_program, kernel_name: &str) -> cl_kernel {
        unimplemented!();
    }
}
