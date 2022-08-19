use metal::ComputePipelineState as MTLPipelineState;
use metal::{ComputeCommandEncoder as MTLComputeCommandEncoder, MTLSize};
use std::sync::{Arc, Mutex};

use crate::Buffer;

pub struct ComputeCommandEncoder {
    pub(crate) encoder: Arc<Mutex<MTLComputeCommandEncoder>>,
}

impl ComputeCommandEncoder {
    pub fn set_buffer(&self, index: usize, buffer: Option<&Buffer>, offset: usize) {
        let locked_encoder = self.encoder.lock().unwrap();
        match buffer {
            Some(buffer) => {
                let locked = buffer.buffer.lock().unwrap();
                locked_encoder.set_buffer(index as u64, Some(locked.as_ref()), offset as u64);
            }
            None => (),
        }
    }

    pub fn set_compute_pipeline_state(&self, pso: &ComputePipelineState) {
        let locked_encoder = self.encoder.lock().unwrap();
        locked_encoder.set_compute_pipeline_state(&pso.pso);
    }

    pub fn dispatch_thread_groups(&self, grid_size: [u32; 3], threadgroup_size: [u32; 3]) {
        let locked_encoder = self.encoder.lock().unwrap();
        let mtl_grid_size = MTLSize::new(
            grid_size[0] as u64,
            grid_size[1] as u64,
            grid_size[2] as u64,
        );
        let mtl_threadgroup_size = MTLSize::new(
            threadgroup_size[0] as u64,
            threadgroup_size[1] as u64,
            threadgroup_size[2] as u64,
        );

        locked_encoder.dispatch_thread_groups(mtl_grid_size, mtl_threadgroup_size);
    }

    pub fn end_encoding(&self) {
        let locked_encoder = self.encoder.lock().unwrap();
        locked_encoder.end_encoding();
    }
}

pub struct ComputePipelineState {
    pub(crate) pso: MTLPipelineState,
}
