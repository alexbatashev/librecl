use crate::common::cl_types::*;
use crate::common::program::Program as CommonProgram;

pub struct Program {}

impl CommonProgram for Program {
    fn get_context() -> cl_context {
        unimplemented!();
    }
}
