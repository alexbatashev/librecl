use super::ClError;
use crate::common::cl_types::ClErrorCode;
use crate::format_error;
use crate::return_error;
use crate::set_info_str;


#[cfg(feature = "vulkan")]
use crate::vulkan;

#[cfg(feature = "metal")]
use crate::metal;

#[cfg(test)]
use crate::mock;


use crate::common::cl_types::cl_context;
use crate::common::cl_types::cl_device_id;
use crate::common::cl_types::cl_int;
use crate::common::cl_types::cl_platform_id;
use crate::common::cl_types::cl_uint;
use crate::common::cl_types::PlatformInfoNames;
use crate::common::cl_types::{CL_INVALID_PLATFORM, CL_INVALID_VALUE, CL_SUCCESS};


use std::rc::Rc;

use crate::lcl_contract;

use super::cl_types::cl_context_callback;
use super::device::ClDevice;

