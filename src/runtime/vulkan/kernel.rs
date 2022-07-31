use super::{SingleDeviceBuffer, SingleDeviceImplicitBuffer};
use crate::api::cl_types::*;
use crate::interface::{KernelImpl, MemKind, ProgramKind};
use crate::interface::{KernelKind, ProgramImpl};
use crate::sync::{self, SharedPtr, UnsafeHandle, WeakPtr};
use librecl_compiler::KernelArgInfo;
use ocl_type_wrapper::ClObjImpl;
use std::ops::Deref;
use std::sync::Arc;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device as VkDevice, DeviceOwned};
use vulkano::pipeline::{ComputePipeline, Pipeline};
use vulkano::VulkanObject;

enum ArgBuffer {
    None,
    SDB(WeakPtr<MemKind>),
    ImplicitBuffer(SingleDeviceImplicitBuffer),
}

#[derive(ClObjImpl)]
pub struct Kernel {
    program: WeakPtr<ProgramKind>,
    name: String,
    args: Vec<KernelArgInfo>,
    arg_buffers: Vec<Arc<ArgBuffer>>,
    #[cl_handle]
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
        return Kernel {
            program,
            name,
            args,
            arg_buffers,
            handle: UnsafeHandle::null(),
        }
        .into();
    }

    // TODO figure out if we actually need to pin memory
    /*
    pub fn pin_memory(&self, device: &Device) {
        for buf in self.arg_buffers {
            match buf {
                ArgBuffer::SDB(_) => (),
                ArgBuffer::ImplicitBuffer(buf) => {

                },
                ArgBuffer::None => panic!(),
            }
        }
    }
    */
    pub fn build_pipeline(
        &self,
        device: Arc<VkDevice>,
    ) -> (Arc<ComputePipeline>, Vec<Arc<PersistentDescriptorSet>>) {
        let owned_program = self.program.upgrade().unwrap();
        let program = match owned_program.deref() {
            ProgramKind::Vulkan(prog) => prog,
            _ => panic!(),
        };
        let pipeline = ComputePipeline::new(
            device,
            program
                .get_module()
                .entry_point(self.name.as_str())
                .unwrap(),
            &(),
            None,
            |_| {},
        )
        .unwrap();

        let mut sets: Vec<Arc<PersistentDescriptorSet>> = vec![];
        let mut wdss: Vec<WriteDescriptorSet> = vec![];
        let layout = pipeline.layout().set_layouts().get(0).unwrap();
        for (idx, arg) in self.arg_buffers.iter().enumerate() {
            let wds = match arg.deref() {
                ArgBuffer::SDB(buffer) => {
                    // TODO pass error
                    let owned_buffer = buffer.upgrade().unwrap();
                    let vk_buf = match owned_buffer.deref() {
                        MemKind::VulkanSDBuffer(buf) => buf,
                        _ => panic!("Unexpected buffer kind"),
                    };

                    WriteDescriptorSet::buffer(idx as u32, vk_buf.get_buffer().clone())
                }
                ArgBuffer::ImplicitBuffer(buffer) => {
                    WriteDescriptorSet::buffer(idx as u32, buffer.get_buffer().clone())
                }
                _ => panic!(),
            };
            wdss.push(wds);
        }
        let set = PersistentDescriptorSet::new(layout.clone(), wdss).unwrap();
        sets.push(set);

        return (pipeline, sets);
    }
}

impl KernelImpl for Kernel {
    fn set_data_arg(&mut self, index: usize, bytes: &[u8]) {
        let owned_program = self.program.upgrade().unwrap();
        let program = match owned_program.deref() {
            ProgramKind::Vulkan(prog) => prog,
            _ => panic!(),
        };
        let buffer = SingleDeviceImplicitBuffer::new(program.get_context(), Vec::from(bytes));
        self.arg_buffers[index] = Arc::new(ArgBuffer::ImplicitBuffer(buffer));
    }
    fn set_buffer_arg(&mut self, index: usize, buffer: WeakPtr<MemKind>) {
        let owned_buffer = buffer.upgrade().unwrap();
        match owned_buffer.deref() {
            MemKind::VulkanSDBuffer(ref vk_buffer) => {
                self.arg_buffers[index] = Arc::new(ArgBuffer::SDB(buffer.clone()));
            }
        }
    }
    fn get_arg_info(&self) -> &[KernelArgInfo] {
        return &self.args;
    }
}
