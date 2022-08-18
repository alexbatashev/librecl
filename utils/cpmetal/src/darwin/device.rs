use metal::Device as MTLDevice;
use std::sync::Arc;
use std::sync::Mutex;

use crate::Buffer;
use crate::CommandQueue;
use crate::ComputePipelineState;
use crate::Function;
use crate::ResourceOptions;
use crate::{CompileOptions, Library};

#[derive(Clone)]
pub struct Device {
    device: Arc<Mutex<MTLDevice>>,
}

impl Device {
    pub fn all() -> Vec<Self> {
        let mut all_devices = vec![];

        let mtl_devices = MTLDevice::all();

        for d in mtl_devices {
            all_devices.push(Device {
                device: Arc::new(Mutex::new(d)),
            });
        }

        return all_devices;
    }

    pub fn name(&self) -> String {
        // TODO deal with poisoned mutex
        let locked_device = self.device.lock().unwrap();
        locked_device.name().to_owned()
    }

    pub fn new_library_with_source(
        &self,
        source: &str,
        options: &CompileOptions,
    ) -> Result<Library, String> {
        let locked_device = self.device.lock().unwrap();
        locked_device
            .new_library_with_source(source, &options.options)
            .map(|lib| Library {
                library: Arc::new(Mutex::new(lib)),
            })
    }

    pub fn new_buffer(&self, size: usize, options: &ResourceOptions) -> Buffer {
        let locked_device = self.device.lock().unwrap();
        Buffer {
            buffer: Arc::new(Mutex::new(
                locked_device.new_buffer(size as u64, options.options),
            )),
        }
    }

    pub fn new_command_queue(&self) -> CommandQueue {
        let locked_device = self.device.lock().unwrap();
        CommandQueue {
            queue: Arc::new(Mutex::new(locked_device.new_command_queue())),
        }
    }

    pub fn new_compute_pipeline_state_with_function(
        &self,
        func: &Function,
    ) -> Result<ComputePipelineState, String> {
        let locked_device = self.device.lock().unwrap();
        let locked_function = func.function.lock().unwrap();
        locked_device
            .new_compute_pipeline_state_with_function(&locked_function)
            .map(|pso| ComputePipelineState { pso })
    }
}

unsafe impl Sync for Device {}
unsafe impl Send for Device {}
