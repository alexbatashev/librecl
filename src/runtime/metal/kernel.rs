use crate::api::cl_types::*;
use crate::interface::{DeviceKind, KernelImpl, KernelKind, MemKind, ProgramKind};
use crate::sync::{self, *};
use cpmetal::{ComputeCommandEncoder, ComputePipelineState, Function, FunctionDescriptor};
use librecl_compiler::KernelArgInfo;
use ocl_type_wrapper::ClObjImpl;
use std::ops::Deref;
use std::sync::Arc;

use super::SingleDeviceBuffer;

enum ArgBuffer {
    None,
    SDB(SingleDeviceBuffer),
}

#[derive(ClObjImpl)]
pub struct Kernel {
    _program: WeakPtr<ProgramKind>,
    _name: String,
    args: Vec<KernelArgInfo>,
    arg_buffers: Vec<Arc<ArgBuffer>>,
    // TODO make per device
    function: Function,
    handle: UnsafeHandle<cl_kernel>,
}

impl Kernel {
    pub fn new(
        program: WeakPtr<ProgramKind>,
        name: String,
        args: Vec<KernelArgInfo>,
    ) -> KernelKind {
        let mut arg_buffers: Vec<Arc<ArgBuffer>> = vec![];
        arg_buffers.resize(args.len(), Arc::new(ArgBuffer::None));

        let descriptor = FunctionDescriptor::new();
        descriptor.set_name(name.as_str());
        let owned_program = program.upgrade().unwrap();
        let function = match owned_program.deref() {
            ProgramKind::Metal(prog) => prog
                .get_library()
                .new_function_with_descriptor(&descriptor)
                .unwrap(),
            #[allow(unreachable_patterns)]
            _ => panic!(),
        };

        Kernel {
            _program: program,
            _name: name,
            args,
            arg_buffers,
            function,
            handle: UnsafeHandle::null(),
        }
        .into()
    }

    pub fn prepare_pso(&self, device: WeakPtr<DeviceKind>) -> ComputePipelineState {
        let owned_device = device.upgrade().unwrap();
        let device_safe = match owned_device.deref() {
            DeviceKind::Metal(device) => device,
            #[allow(unreachable_patterns)]
            _ => panic!(),
        };

        let native_device = device_safe.get_native_device();

        return native_device
            .new_compute_pipeline_state_with_function(&self.function)
            .unwrap();
    }

    pub fn encode_arguments(&self, compute_encoder: &ComputeCommandEncoder) {
        for (idx, arg) in self.arg_buffers.iter().enumerate() {
            match arg.as_ref() {
                ArgBuffer::SDB(ref buffer) => buffer.encode_argument(compute_encoder, idx),
                #[allow(unreachable_patterns)]
                _ => panic!(),
            };
        }
    }
}

impl KernelImpl for Kernel {
    fn set_data_arg(&mut self, _index: usize, _bytes: &[u8]) {
        unimplemented!();
    }
    fn set_buffer_arg(&mut self, index: usize, buffer: WeakPtr<MemKind>) {
        let owned_buffer = buffer.upgrade().unwrap();
        match owned_buffer.deref() {
            MemKind::MetalSDBuffer(ref buffer) => {
                self.arg_buffers[index] = Arc::new(ArgBuffer::SDB(buffer.clone()));
            }
            #[allow(unreachable_patterns)]
            _ => panic!(),
        }
    }
    fn get_arg_info(&self) -> &[KernelArgInfo] {
        return &self.args;
    }
}
