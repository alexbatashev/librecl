use metal::Buffer as MTLBuffer;
use metal::MTLResourceOptions;
use std::sync::Arc;
use std::sync::Mutex;

pub struct ResourceOptions {
  pub(crate) options: MTLResourceOptions,
}

impl ResourceOptions {
  pub fn empty() -> ResourceOptions {
    ResourceOptions {
      options: MTLResourceOptions::empty(),
    }
  }
}

#[derive(Clone)]
pub struct Buffer {
  pub(crate) buffer: Arc<Mutex<MTLBuffer>>,
}

impl Buffer {
  pub fn contents(&self) -> *mut libc::c_void {
    let locked_buffer = self.buffer.lock().unwrap();
    locked_buffer.contents()
  }
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}
