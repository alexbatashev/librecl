use crate::common::cl_types::cl_context;
use crate::common::cl_types::cl_context_callback;
use crate::common::cl_types::cl_device_id;
use crate::common::context::ClContext;
use crate::common::context::Context as CommonContext;

pub struct Context {
    devices: Vec<cl_device_id>,
    error_callback: cl_context_callback,
    callback_user_data: *mut libc::c_void,
}

impl Context {
    pub fn new(
        devices: &[cl_device_id],
        error_callback: cl_context_callback,
        callback_user_data: *mut libc::c_void,
    ) -> cl_context {
        let mut owned_devices = vec![];
        owned_devices.extend_from_slice(devices);
        let ctx = Box::into_raw(Box::<ClContext>::new(
            Context {
                devices: owned_devices,
                error_callback,
                callback_user_data,
            }
            .into(),
        ));
        return ctx;
    }
}

impl CommonContext for Context {
    fn notify_error(&self, message: String) {
        unimplemented!();
    }
    fn has_device(&self, device: cl_device_id) -> bool {
        return self.devices.contains(&device);
    }
}
