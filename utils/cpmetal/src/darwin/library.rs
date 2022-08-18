use metal::CompileOptions as MTLCompileOptions;
use metal::Library as MTLLibrary;
use std::sync::{Arc, Mutex};

use crate::Function;
use crate::FunctionDescriptor;

pub struct CompileOptions {
    pub(crate) options: MTLCompileOptions,
}

impl CompileOptions {
    pub fn new() -> CompileOptions {
        CompileOptions {
            options: MTLCompileOptions::new(),
        }
    }
}

#[derive(Clone)]
pub struct Library {
    pub(crate) library: Arc<Mutex<MTLLibrary>>,
}

impl Library {
    pub fn new_function_with_descriptor(
        &self,
        descriptor: &FunctionDescriptor,
    ) -> Result<Function, String> {
        let locked_library = self.library.lock().unwrap();
        locked_library
            .new_function_with_descriptor(&descriptor.descriptor)
            .map(|func| Function {
                function: Arc::new(Mutex::new(func)),
            })
    }
}
