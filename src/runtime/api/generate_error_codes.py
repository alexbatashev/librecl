errors = """#define CL_SUCCESS                                  0
#define CL_DEVICE_NOT_FOUND                         -1
#define CL_DEVICE_NOT_AVAILABLE                     -2
#define CL_COMPILER_NOT_AVAILABLE                   -3
#define CL_MEM_OBJECT_ALLOCATION_FAILURE            -4
#define CL_OUT_OF_RESOURCES                         -5
#define CL_OUT_OF_HOST_MEMORY                       -6
#define CL_PROFILING_INFO_NOT_AVAILABLE             -7
#define CL_MEM_COPY_OVERLAP                         -8
#define CL_IMAGE_FORMAT_MISMATCH                    -9
#define CL_IMAGE_FORMAT_NOT_SUPPORTED               -10
#define CL_BUILD_PROGRAM_FAILURE                    -11
#define CL_MAP_FAILURE                              -12
#define CL_MISALIGNED_SUB_BUFFER_OFFSET             -13
#define CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST -14
#define CL_COMPILE_PROGRAM_FAILURE                  -15
#define CL_LINKER_NOT_AVAILABLE                     -16
#define CL_LINK_PROGRAM_FAILURE                     -17
#define CL_DEVICE_PARTITION_FAILED                  -18
#define CL_KERNEL_ARG_INFO_NOT_AVAILABLE            -19
#define CL_INVALID_VALUE                            -30
#define CL_INVALID_DEVICE_TYPE                      -31
#define CL_INVALID_PLATFORM                         -32
#define CL_INVALID_DEVICE                           -33
#define CL_INVALID_CONTEXT                          -34
#define CL_INVALID_QUEUE_PROPERTIES                 -35
#define CL_INVALID_COMMAND_QUEUE                    -36
#define CL_INVALID_HOST_PTR                         -37
#define CL_INVALID_MEM_OBJECT                       -38
#define CL_INVALID_IMAGE_FORMAT_DESCRIPTOR          -39
#define CL_INVALID_IMAGE_SIZE                       -40
#define CL_INVALID_SAMPLER                          -41
#define CL_INVALID_BINARY                           -42
#define CL_INVALID_BUILD_OPTIONS                    -43
#define CL_INVALID_PROGRAM                          -44
#define CL_INVALID_PROGRAM_EXECUTABLE               -45
#define CL_INVALID_KERNEL_NAME                      -46
#define CL_INVALID_KERNEL_DEFINITION                -47
#define CL_INVALID_KERNEL                           -48
#define CL_INVALID_ARG_INDEX                        -49
#define CL_INVALID_ARG_VALUE                        -50
#define CL_INVALID_ARG_SIZE                         -51
#define CL_INVALID_KERNEL_ARGS                      -52
#define CL_INVALID_WORK_DIMENSION                   -53
#define CL_INVALID_WORK_GROUP_SIZE                  -54
#define CL_INVALID_WORK_ITEM_SIZE                   -55
#define CL_INVALID_GLOBAL_OFFSET                    -56
#define CL_INVALID_EVENT_WAIT_LIST                  -57
#define CL_INVALID_EVENT                            -58
#define CL_INVALID_OPERATION                        -59
#define CL_INVALID_GL_OBJECT                        -60
#define CL_INVALID_BUFFER_SIZE                      -61
#define CL_INVALID_MIP_LEVEL                        -62
#define CL_INVALID_GLOBAL_WORK_SIZE                 -63
#define CL_INVALID_PROPERTY                         -64
#define CL_INVALID_IMAGE_DESCRIPTOR                 -65
#define CL_INVALID_COMPILER_OPTIONS                 -66
#define CL_INVALID_LINKER_OPTIONS                   -67
#define CL_INVALID_DEVICE_PARTITION_COUNT           -68
#define CL_INVALID_PIPE_SIZE                        -69
#define CL_INVALID_DEVICE_QUEUE                     -70
#define CL_INVALID_SPEC_ID                          -71
#define CL_MAX_SIZE_RESTRICTION_EXCEEDED            -72"""

lines = errors.splitlines()

enum = """#[derive(Debug, Clone)]
pub enum ClError {
"""

impl = """impl ClError {
    pub fn error_code(&self) -> cl_int {
        match self {
"""

get_name = """
    pub fn get_name(&self) -> &str {
        match self {
"""

fmt = """impl fmt::Display for ClError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let error_descr = match self {
"""

test = """#[cfg(test)]
pub mod error_codes {
"""

for line in lines:
    parts = line.split(" ")
    enum_name = parts[1].replace("CL_", "")
    enum_name = ''.join(x.title() for x in enum_name.split("_"))
    enum = enum + "    " + enum_name + "(ErrorDescription),\n"
    impl = impl + "            ClError::" + enum_name + "(_) => " + parts[-1] + " as cl_int,\n"
    get_name = get_name + "            ClError::" + enum_name + "(_) => \"" + parts[1] + "\",\n"
    fmt = fmt + "            ClError::" + enum_name + "(desc) => desc,\n"
    test = test + "    pub const " + parts[1] + ": cl_int = " + parts[-1] + ";\n"

enum = enum + "}\n"
impl = impl + "        }\n    }\n"
get_name = get_name + "        }\n    }\n}\n"
fmt = fmt + """        };
        write!(f, "{} ({}) - {} from:\\n{:?}", self.error_code(), self.get_name(), error_descr.message, error_descr.backtrace)
    }
}"""
test = test + "}\n"

print(enum)
print(impl)
print(get_name)
print(fmt)
print(test)
