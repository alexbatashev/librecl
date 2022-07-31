use crate::api::cl_types::*;
use crate::interface::{DeviceKind, KernelImpl, KernelKind, MemKind, ProgramKind};
use crate::sync::{self, *};
use librecl_compiler::{KernelArgInfo, KernelArgType};
use metal_api::{
    ComputeCommandEncoder, ComputeCommandEncoderRef, ComputePipelineState, Function,
    FunctionDescriptor,
};
use ocl_type_wrapper::ClObjImpl;
use std::ops::Deref;
use std::sync::{Arc, Mutex};

use super::SingleDeviceBuffer;

enum ArgBuffer {
    None,
    SDB(SingleDeviceBuffer),
}

#[derive(ClObjImpl)]
pub struct Kernel {
    program: WeakPtr<ProgramKind>,
    name: String,
    args: Vec<KernelArgInfo>,
    arg_buffers: Vec<Arc<ArgBuffer>>,
    // TODO make per device
    function: Arc<Mutex<UnsafeHandle<Function>>>,
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
        let function = Arc::new(Mutex::new(UnsafeHandle::new(match owned_program.deref() {
            ProgramKind::Metal(prog) => prog
                .get_library()
                .new_function_with_descriptor(&descriptor)
                .unwrap(),
            _ => panic!(),
        })));

        Kernel {
            program,
            name,
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
            _ => panic!(),
        };

        let locked_function = self.function.lock().unwrap();
        let locked_device = device_safe.get_native_device().lock().unwrap();

        return locked_device
            .value()
            .new_compute_pipeline_state_with_function(locked_function.value())
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

impl KernelImpl for Kernel {
    fn set_data_arg(&mut self, index: usize, bytes: &[u8]) {
        unimplemented!();
    }
    fn set_buffer_arg(&mut self, index: usize, buffer: WeakPtr<MemKind>) {
        let owned_buffer = buffer.upgrade().unwrap();
        match owned_buffer.deref() {
            MemKind::MetalSDBuffer(ref buffer) => {
                self.arg_buffers[index] = Arc::new(ArgBuffer::SDB(buffer.clone()));
            }
        }
    }
    fn get_arg_info(&self) -> &[KernelArgInfo] {
        return &self.args;
    }
}
