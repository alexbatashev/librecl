use super::Kernel;
use crate::api::cl_types::*;
use crate::interface::{ContextKind, DeviceKind, KernelKind, ProgramImpl, ProgramKind};
use crate::sync::{self, SharedPtr, UnsafeHandle, WeakPtr};
use librecl_compiler::FrontendResult;
use librecl_compiler::{Backend, KernelInfo};
use ocl_type_wrapper::ClObjImpl;
use std::ops::Deref;
use std::sync::Arc;
use vulkano::shader::ShaderModule;

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
    module: Option<Arc<ShaderModule>>,
    #[cl_handle]
    handle: UnsafeHandle<cl_program>,
}

impl Program {
    pub fn new(context: WeakPtr<ContextKind>, program_content: ProgramContent) -> Program {
        return Program {
            context,
            program_content,
            frontend_result: Option::None,
            kernels: vec![],
            binary: vec![],
            module: Option::None,
            handle: UnsafeHandle::null(),
        };
    }

    pub fn get_module(&self) -> &ShaderModule {
        return &self.module.as_ref().unwrap();
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
            ContextKind::Vulkan(vk_ctx) => vk_ctx,
            #[allow(unreachable_patterns)]
            _ => panic!("Unsupported enum value"),
        };
        let compile_result = match &mut self.program_content {
            ProgramContent::Source(source) => {
                let cfe = context.get_clang_fe();
                let result = cfe.process_source(source.as_str());
                Some(result)
            }
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
            ContextKind::Vulkan(vk_ctx) => vk_ctx,
            #[allow(unreachable_patterns)]
            _ => panic!("Unsupported enum value"),
        };

        let be = context.get_vulkan_be();
        let result = be.compile(&self.frontend_result.as_ref().unwrap());

        self.kernels = result.get_kernels();
        self.binary = result.get_binary();
        // TODO support multiple devices.
        let owned_device = devices[0].upgrade().unwrap();
        let device = match owned_device.deref() {
            DeviceKind::Vulkan(device) => device.get_logical_device(),
            _ => panic!("Unexpected enum value"),
        };
        // TODO handle error
        self.module = Some(
            unsafe {
                ShaderModule::from_words(
                    device.clone(),
                    bytemuck::cast_slice(self.binary.as_slice()),
                )
            }
            .unwrap(),
        );
        self.frontend_result = Option::None;

        // TODO how to handle compiler errors?
        return true;
    }

    fn create_kernel(&self, kernel_name: &str) -> KernelKind {
        let maybe_kernel: Option<&KernelInfo> =
            (&self.kernels).into_iter().find(|&k| k.name == kernel_name);
        let owned_program: SharedPtr<ProgramKind> =
            FromCl::try_from_cl(*self.handle.value()).unwrap();
        match maybe_kernel {
            Some(kernel_info) => Kernel::new(
                SharedPtr::downgrade(&owned_program),
                kernel_info.name.clone(),
                kernel_info.arguments.clone(),
            )
            .into(),
            _ => {
                // TODO this should not have happened
                // Consider more diagnostics
                panic!("No kernel named {}", kernel_name);
            }
        }
    }
}
