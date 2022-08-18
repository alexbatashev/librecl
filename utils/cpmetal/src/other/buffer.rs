pub struct ResourceOptions {}

#[allow(dead_code)]
impl ResourceOptions {
    pub fn empty() -> ResourceOptions {
        unimplemented!()
    }
}

#[derive(Clone)]
pub struct Buffer {}

#[allow(dead_code)]
impl Buffer {
    pub fn contents(&self) -> *mut libc::c_void {
        unimplemented!()
    }
}
