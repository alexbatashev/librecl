use super::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Function, Library, ResourceOptions,
};

pub struct Device {}

#[allow(dead_code)]
impl Device {
    pub fn all() -> Vec<Self> {
        return vec![];
    }

    pub fn name(&self) -> String {
        unimplemented!()
    }

    pub fn new_library_with_source(
        &self,
        _source: &str,
        _options: &CompileOptions,
    ) -> Result<Library, String> {
        unimplemented!()
    }

    pub fn new_buffer(&self, _size: usize, _options: &ResourceOptions) -> Buffer {
        unimplemented!()
    }

    pub fn new_command_queue(&self) -> CommandQueue {
        unimplemented!()
    }

    pub fn new_compute_pipeline_state_with_function(
        &self,
        _func: &Function,
    ) -> Result<ComputePipelineState, String> {
        unimplemented!()
    }
}
