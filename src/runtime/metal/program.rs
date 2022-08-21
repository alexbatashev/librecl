use crate::api::cl_types::*;
use crate::interface::{ContextKind, DeviceKind, KernelKind, ProgramImpl, ProgramKind};
use crate::sync::{self, *};
use cpmetal::{CompileOptions, Library};
use librecl_compiler::CompileResult;
use librecl_compiler::KernelInfo;
use ocl_args::parse_options;
use ocl_type_wrapper::ClObjImpl;
use std::ops::Deref;
use std::sync::Arc;

use super::Kernel;

pub enum ProgramContent {
    Source(String),
    SPIRV(Vec<i8>),
}

#[derive(ClObjImpl)]
pub struct Program {
    context: WeakPtr<ContextKind>,
    program_content: ProgramContent,
    compile_result: Option<Arc<CompileResult>>,
    kernels: Vec<KernelInfo>,
    _binary: Vec<u8>,
    library: Option<Library>,
    handle: UnsafeHandle<cl_program>,
}

impl Program {
    pub fn new(context: WeakPtr<ContextKind>, program_content: ProgramContent) -> ProgramKind {
        Program {
            context,
            program_content,
            compile_result: Option::None,
            kernels: vec![],
            _binary: vec![],
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
    fn compile_program(&mut self, devices: &[WeakPtr<DeviceKind>]) -> bool {
        let owned_device = devices.first().unwrap().upgrade().unwrap();
        let device = match owned_device.deref() {
            DeviceKind::Metal(device) => device,
            #[allow(unreachable_patterns)]
            _ => panic!("unsupported enum"),
        };
        let compile_result = match &mut self.program_content {
            ProgramContent::Source(source) => {
                let split_options: [String; 2] =
                    [String::from("-c"), String::from("--targets=metal-macos")];
                let options = parse_options(&split_options).expect("options");
                let result = device
                    .get_compiler()
                    .compile_source(source.as_str(), &options);
                Some(result)
            }
            ProgramContent::SPIRV(spirv) => {
                // TODO pass spec constants
                let split_options: [String; 2] =
                    [String::from("-c"), String::from("--targets=metal-macos")];
                let options = parse_options(&split_options).expect("options");
                let result = device.get_compiler().compile_spirv(spirv, &options);
                Some(result)
            }
        };

        self.compile_result = compile_result;

        return self.compile_result.is_some() && self.compile_result.as_ref().unwrap().is_ok();
    }
    fn link_programs(&mut self, devices: &[WeakPtr<DeviceKind>]) -> bool {
        if !self.compile_result.is_some() {
            return false;
        }

        let owned_device = devices.first().unwrap().upgrade().unwrap();
        let device = match owned_device.deref() {
            DeviceKind::Metal(device) => device,
            #[allow(unreachable_patterns)]
            _ => panic!("unsupported enum"),
        };

        let compiler = device.get_compiler();
        let mut modules = vec![self.compile_result.as_ref().unwrap().clone()];
        for lib in device.get_builtin_libs() {
            modules.push(lib.clone());
        }

        let split_options: [String; 1] = [String::from("--targets=metal-macos")];
        let options = parse_options(&split_options).expect("options");
        let result = compiler.link(&modules, &options);

        self.kernels = result.get_kernels();
        let msl = result.get_binary();
        // TODO support multiple devices.
        let owned_device = devices[0].upgrade().unwrap();
        let device = match owned_device.deref() {
            DeviceKind::Metal(device) => device.get_native_device(),
            _ => panic!("Unexpected enum value"),
        };
        // TODO proper error handling
        let options = CompileOptions::new();
        self.library = Some(
            device
                .new_library_with_source(std::str::from_utf8(&msl).unwrap(), &options)
                .unwrap(),
        );

        self.compile_result = None;

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
