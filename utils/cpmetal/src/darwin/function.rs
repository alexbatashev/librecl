use std::sync::{Arc, Mutex};

use metal::FunctionDescriptor as MTLFunctionDescriptor;
use metal::Function as MTLFunction;

pub struct FunctionDescriptor {
  pub(crate) descriptor: MTLFunctionDescriptor,
}

impl FunctionDescriptor {
  pub fn new() -> FunctionDescriptor {
    FunctionDescriptor { descriptor: MTLFunctionDescriptor::new(), }
  }

  pub fn set_name(&self, name: &str) {
    self.descriptor.set_name(name);
  }
}

#[derive(Clone)]
pub struct Function {
  pub(crate) function: Arc<Mutex<MTLFunction>>,
}

unsafe impl Sync for Function {}
unsafe impl Send for Function {}
