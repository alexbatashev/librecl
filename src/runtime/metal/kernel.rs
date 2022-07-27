use crate::common::device::ClDevice;
use crate::common::kernel::Kernel as CommonKernel;
use crate::common::memory::ClMem;
use crate::common::{cl_types::*, program::ClProgram};
use librecl_compiler::{KernelArgInfo, KernelArgType};
use metal_api::{
    ComputeCommandEncoder, ComputeCommandEncoderRef, ComputePipelineState, Function,
    FunctionDescriptor,
};
use std::sync::Arc;

use super::SingleDeviceBuffer;

enum ArgBuffer {
    None,
    SDB(SingleDeviceBuffer),
}

pub struct Kernel {
    program: cl_program,
    name: String,
    args: Vec<KernelArgInfo>,
    arg_buffers: Vec<Arc<ArgBuffer>>,
    // TODO make per device
    function: Arc<Function>,
}

impl Kernel {
    pub fn new(program: cl_program, name: String, args: Vec<KernelArgInfo>) -> cl_kernel {
        let mut arg_buffers: Vec<Arc<ArgBuffer>> = vec![];
        arg_buffers.resize(args.len(), Arc::new(ArgBuffer::None));

        let descriptor = FunctionDescriptor::new();
        descriptor.set_name(name.as_str());
        let function = Arc::new(match unsafe { program.as_ref() }.unwrap() {
            ClProgram::Metal(prog) => prog
                .get_library()
                .new_function_with_descriptor(&descriptor)
                .unwrap(),
            _ => panic!(),
        });

        return Box::into_raw(Box::new(
            Kernel {
                program,
                name,
                args,
                arg_buffers,
                function,
            }
            .into(),
        ));
    }

    pub fn prepare_pso(&self, device: cl_device_id) -> ComputePipelineState {
        let device_safe = match unsafe { device.as_ref() }.unwrap() {
            ClDevice::Metal(device) => device,
            _ => panic!(),
        };

        return device_safe
            .get_native_device()
            .new_compute_pipeline_state_with_function(&self.function)
            .unwrap();
    }

    pub fn encode_arguments(&self, compute_encoder: &ComputeCommandEncoderRef) {
        for (idx, arg) in self.arg_buffers.iter().enumerate() {
            match arg.as_ref() {
                ArgBuffer::SDB(ref buffer) => buffer.encode_argument(compute_encoder, idx),
                _ => panic!(),
            };
        }
    }
}

impl CommonKernel for Kernel {
    fn set_data_arg(&mut self, index: usize, bytes: &[u8]) {
        unimplemented!();
    }
    fn set_buffer_arg(&mut self, index: usize, buffer: cl_mem) {
        match unsafe { buffer.as_ref() }.unwrap() {
            ClMem::MetalSDBuffer(ref buffer) => {
                self.arg_buffers[index] = Arc::new(ArgBuffer::SDB(buffer.clone()));
            }
        }
    }
    fn get_arg_info(&self) -> &[KernelArgInfo] {
        return &self.args;
    }
}
