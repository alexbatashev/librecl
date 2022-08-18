#[derive(Debug, Clone)]
#[repr(u32)]
pub enum ContextInfoNames {
    ReferenceCount = 0x1080,
    Devices = 0x1081,
    Properties = 0x1082,
    NumDevices = 0x1083,
}

impl ContextInfoNames {
    pub fn as_cl_str(&self) -> &str {
        match self {
            ContextInfoNames::ReferenceCount => "CL_CONTEXT_REFERENCE_COUNT",
            ContextInfoNames::Devices => "CL_CONTEXT_DEVICES",
            ContextInfoNames::Properties => "CL_CONTEXT_PROPERTIES",
            ContextInfoNames::NumDevices => "CL_CONTEXT_NUM_DEVICES",
        }
    }
}

impl TryFrom<cl_uint> for ContextInfoNames {
    type Error = ();
    fn try_from(v: cl_uint) -> Result<Self, Self::Error> {
        match v {
            x if x == ContextInfoNames::ReferenceCount as cl_uint => Ok(ContextInfoNames::ReferenceCount),
            x if x == ContextInfoNames::Devices as cl_uint => Ok(ContextInfoNames::Devices),
            x if x == ContextInfoNames::Properties as cl_uint => Ok(ContextInfoNames::Properties),
            x if x == ContextInfoNames::NumDevices as cl_uint => Ok(ContextInfoNames::NumDevices),
            _ => Err(()),
        }
    }
}

