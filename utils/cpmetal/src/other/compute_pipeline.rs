use super::Buffer;

pub struct ComputeCommandEncoder {}

#[allow(dead_code)]
impl ComputeCommandEncoder {
    pub fn set_buffer(&self, _index: usize, _buffer: Option<&Buffer>, _offset: usize) {
        unimplemented!()
    }

    pub fn set_compute_pipeline_state(&self, _pso: &ComputePipelineState) {
        unimplemented!()
    }

    pub fn dispatch_thread_groups(&self, _grid_size: [u32; 3], _threadgroup_size: [u32; 3]) {
        unimplemented!()
    }

    pub fn end_encoding(&self) {
        unimplemented!()
    }
}

pub struct ComputePipelineState {}
