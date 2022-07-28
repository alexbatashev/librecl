use crate::common::cl_types::*;
use crate::common::context::ClContext;
use crate::common::device::ClDevice;
use crate::common::program::Program as CommonProgram;
use librecl_compiler::{Backend, KernelInfo};
use librecl_compiler::{BinaryProgram, FrontendResult};
use metal_api::{CompileOptions, Library};

use super::Kernel;

pub enum ProgramContent {
    Source(String),
}

pub struct Program {
    context: cl_context,
    program_content: ProgramContent,
    frontend_result: Option<FrontendResult>,
    kernels: Vec<KernelInfo>,
    binary: Vec<u8>,
    library: Option<Library>,
}

impl Program {
    pub fn new(context: cl_context, program_content: ProgramContent) -> cl_program {
        return Box::into_raw(Box::new(
            Program {
                context,
                program_content,
                frontend_result: Option::None,
                kernels: vec![],
                binary: vec![],
                library: Option::None,
            }
            .into(),
        ));
    }
    pub fn get_library(&self) -> &Library {
        return self.library.as_ref().unwrap();
    }
}

unsafe impl Sync for Program {}
unsafe impl Send for Program {}

impl CommonProgram for Program {
    fn get_context(&self) -> cl_context {
        return self.context;
    }
    fn get_safe_context_mut<'a, 'b>(&'a mut self) -> &'b mut ClContext {
        return unsafe { self.context.as_mut() }.unwrap();
    }
    fn compile_program(&mut self, _devices: &[&ClDevice]) -> bool {
        // TODO do we need devices here?
        let context = match self.get_safe_context_mut() {
            ClContext::Metal(ctx) => ctx,
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
    fn link_programs(&mut self, devices: &[&ClDevice]) -> bool {
        if !self.frontend_result.is_some() {
            return false;
        }

        let context = match self.get_safe_context_mut() {
            ClContext::Metal(ctx) => ctx,
            _ => panic!("Unsupported enum value"),
        };

        let be = context.get_metal_be();
        let result = be.compile(&self.frontend_result.as_ref().unwrap());

        self.kernels = result.get_kernels();
        let msl = result.get_binary();
        // TODO support multiple devices.
        let device = match devices[0] {
            ClDevice::Metal(device) => device.get_native_device(),
        };
        // TODO proper error handling
        let options = CompileOptions::new();
        self.library = Some(
            device
                .new_library_with_source(std::str::from_utf8(&msl).unwrap(), &options)
                .unwrap(),
        );

        self.frontend_result = None;

        return true;
    }

    fn create_kernel(&self, program: cl_program, kernel_name: &str) -> cl_kernel {
        let maybe_kernel: Option<&KernelInfo> =
            (&self.kernels).into_iter().find(|&k| k.name == kernel_name);
        match maybe_kernel {
            Some(kernel_info) => Kernel::new(
                program,
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
