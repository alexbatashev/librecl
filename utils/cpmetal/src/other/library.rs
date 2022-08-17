use super::{Function, FunctionDescriptor};

pub struct CompileOptions {}

#[allow(dead_code)]
impl CompileOptions {
    pub fn new() -> CompileOptions {
        unimplemented!()
    }
}

#[derive(Clone)]
pub struct Library {}

#[allow(dead_code)]
impl Library {
    pub fn new_function_with_descriptor(
        &self,
        _descriptor: &FunctionDescriptor,
    ) -> Result<Function, String> {
        unimplemented!()
    }
}
