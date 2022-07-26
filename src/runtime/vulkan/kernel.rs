use librecl_compiler::KernelArgInfo;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{ComputePipeline, Pipeline};
use vulkano::VulkanObject;

use super::{SingleDeviceBuffer, SingleDeviceImplicitBuffer};
use crate::common::kernel::Kernel as CommonKernel;
use crate::common::memory::ClMem;
use crate::common::program::Program as CommpnProgram;
use crate::common::{cl_types::*, program::ClProgram};
use std::ops::Deref;
use std::sync::Arc;
use vulkano::device::{Device as VkDevice, DeviceOwned};

enum ArgBuffer {
    None,
    SDB(SingleDeviceBuffer),
    ImplicitBuffer(SingleDeviceImplicitBuffer),
}

pub struct Kernel {
    program: cl_program,
    name: String,
    args: Vec<KernelArgInfo>,
    arg_buffers: Vec<Arc<ArgBuffer>>,
}

impl Kernel {
    pub fn new(program: cl_program, name: String, args: Vec<KernelArgInfo>) -> Kernel {
        let mut arg_buffers: Vec<Arc<ArgBuffer>> = vec![];
        arg_buffers.resize(args.len(), Arc::new(ArgBuffer::None));
        return Kernel {
            program,
            name,
            args,
            arg_buffers,
        };
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
        let program = match unsafe { self.program.as_ref() }.unwrap() {
            ClProgram::Vulkan(prog) => prog,
            _ => panic!(),
        };
        mod cs {
            vulkano_shaders::shader! {
                ty: "compute",
                src: "
                    #version 450

                    layout (std430, set=0, binding=0) buffer inA { int a[]; };
                    layout (std430, set=0, binding=1) buffer inB { int b[]; };
                    layout (std430, set=0, binding=2) buffer outR { int result[]; };

                    void main() {
                      const uint i = gl_GlobalInvocationID.x;
                      result[i] = a[i] + b[i];
                    }
                "
            }
        }
        let shader = cs::load(device.clone()).unwrap();
        let pipeline = ComputePipeline::new(
            device,
            // shader.entry_point("main").unwrap(),
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
                    let buf = buffer.get_buffer();
                    buf.device();
                    buf.device().internal_object();
                    WriteDescriptorSet::buffer(idx as u32, buffer.get_buffer().clone())
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

impl CommonKernel for Kernel {
    fn set_data_arg(&mut self, index: usize, bytes: &[u8]) {
        let program = match unsafe { self.program.as_ref() }.unwrap() {
            ClProgram::Vulkan(prog) => prog,
            _ => panic!(),
        };
        let buffer = SingleDeviceImplicitBuffer::new(program.get_context(), Vec::from(bytes));
        self.arg_buffers[index] = Arc::new(ArgBuffer::ImplicitBuffer(buffer));
    }
    fn set_buffer_arg(&mut self, index: usize, buffer: cl_mem) {
        match unsafe { buffer.as_ref() }.unwrap() {
            ClMem::VulkanSDBuffer(ref vk_buffer) => {
                vk_buffer.get_buffer().device().internal_object();
                self.arg_buffers[index] = Arc::new(ArgBuffer::SDB(vk_buffer.clone()));
            }
        }
    }
    fn get_arg_info(&self) -> &[KernelArgInfo] {
        return &self.args;
    }
}
