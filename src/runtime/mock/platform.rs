use crate::common::cl_types::*;
use crate::common::device::ClDevice;
use crate::common::platform::ClPlatform;
use crate::common::platform::Platform as CommonPlatform;
use std::rc::Rc;
use std::sync::Arc;

pub struct Platform {}

impl Platform {
    pub fn create_platforms(platforms: &mut Vec<Arc<ClPlatform>>) {
        platforms.push(Arc::new(Platform {}.into()));
    }
}

impl CommonPlatform for Platform {
    fn get_platform_name(&self) -> &str {
        unimplemented!();
    }
    fn get_devices(&self) -> &Vec<Rc<ClDevice>> {
        unimplemented!();
    }
    fn add_device(&mut self, device: Rc<ClDevice>) {
        unimplemented!();
    }
    fn create_context(
        &self,
        devices: &[cl_device_id],
        callback: cl_context_callback,
        user_data: *mut libc::c_void,
    ) -> cl_context {
        unimplemented!();
    }
}
