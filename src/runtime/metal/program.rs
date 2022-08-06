use std::ops::Deref;

use crate::api::cl_types::*;
use crate::interface::{ContextKind, DeviceKind, KernelKind, ProgramImpl, ProgramKind};
use crate::sync::{self, *};
use librecl_compiler::{Backend, KernelInfo};
use librecl_compiler::{BinaryProgram, FrontendResult};
use metal_api::{CompileOptions, Library};
use ocl_type_wrapper::ClObjImpl;

use super::Kernel;

pub enum ProgramContent {
    Source(String),
}

#[derive(ClObjImpl)]
pub struct Program {
    context: WeakPtr<ContextKind>,
    program_content: ProgramContent,
    frontend_result: Option<FrontendResult>,
    kernels: Vec<KernelInfo>,
    binary: Vec<u8>,
    library: Option<Library>,
    handle: UnsafeHandle<cl_program>,
}

impl Program {
    pub fn new(context: WeakPtr<ContextKind>, program_content: ProgramContent) -> ProgramKind {
        Program {
            context,
            program_content,
            frontend_result: Option::None,
            kernels: vec![],
            binary: vec![],
            library: Option::None,
            handle: UnsafeHandle::null(),
        }
        .into()
    }
    pub fn get_library(&self) -> &Library {
        return self.library.as_ref().unwrap();
    }
}

unsafe impl Sync for Program {}
unsafe impl Send for Program {}

impl ProgramImpl for Program {
    fn get_context(&self) -> WeakPtr<ContextKind> {
        return self.context.clone();
    }
    fn compile_program(&mut self, _devices: &[WeakPtr<DeviceKind>]) -> bool {
        // TODO do we need devices here?
        let owned_context = self.context.upgrade().unwrap();
        let context = match owned_context.deref() {
            ContextKind::Metal(ctx) => ctx,
            #[allow(unreachable_patterns)]
            _ => panic!("Unsupported enum value"),
        };
        let compile_result = match &mut self.program_content {
            ProgramContent::Source(source) => {
                let cfe = context.get_clang_fe();
                let result = cfe.process_source(source.as_str());
                Some(result)
            }
            _ => None,
        };

        self.frontend_result = compile_result;

        return self.frontend_result.is_some() && self.frontend_result.as_ref().unwrap().is_ok();
    }
    fn link_programs(&mut self, devices: &[WeakPtr<DeviceKind>]) -> bool {
        if !self.frontend_result.is_some() {
            return false;
        }

        let owned_context = self.context.upgrade().unwrap();
        let context = match owned_context.deref() {
            ContextKind::Metal(ctx) => ctx,
            #[allow(unreachable_patterns)]
            _ => panic!("Unsupported enum value"),
        };

        let be = context.get_metal_be();
        let result = be.compile(&self.frontend_result.as_ref().unwrap());

        self.kernels = result.get_kernels();
        let msl = result.get_binary();
        // TODO support multiple devices.
        let owned_device = devices[0].upgrade().unwrap();
        let device = match owned_device.deref() {
            DeviceKind::Metal(device) => device.get_native_device(),
        };
        // TODO proper error handling
        let options = CompileOptions::new();
        let locked_device = device.lock().unwrap();
        self.library = Some(
            locked_device
                .value()
                .new_library_with_source(std::str::from_utf8(&msl).unwrap(), &options)
                .unwrap(),
        );

        self.frontend_result = None;

        return true;
    }

    fn create_kernel(&self, kernel_name: &str) -> KernelKind {
        let maybe_kernel: Option<&KernelInfo> =
            (&self.kernels).into_iter().find(|&k| k.name == kernel_name);
        let owned_program = ProgramKind::try_from_cl(*self.handle.value()).unwrap();
        match maybe_kernel {
            Some(kernel_info) => Kernel::new(
                SharedPtr::downgrade(&owned_program),
                kernel_info.name.clone(),
                kernel_info.arguments.clone(),
            ),
            _ => {
                // TODO this should not have happened
                // Consider more diagnostics
                panic!("No kernel named {}", kernel_name);
            }
        }
    }
}
