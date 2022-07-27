use crate::common::cl_types::*;
use crate::common::queue::Queue as CommonQueue;

pub struct Queue {}

impl CommonQueue for Queue {
    fn enqueue_buffer_write(&self, src: *const libc::c_void, dst: cl_mem) {
        unimplemented!();
    }
    fn enqueue_buffer_read(&self, src: cl_mem, dst: *mut libc::c_void) {
        unimplemented!();
    }
    fn submit(
        &self,
        kernel: cl_kernel,
        offset: [u32; 3],
        global_size: [u32; 3],
        local_size: [u32; 3],
    ) {
        unimplemented!();
    }
    fn finish(&self) {
        unimplemented!();
    }
}
