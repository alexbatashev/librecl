info_names = """#define CL_CONTEXT_REFERENCE_COUNT                  0x1080
#define CL_CONTEXT_DEVICES                          0x1081
#define CL_CONTEXT_PROPERTIES                       0x1082
#define CL_CONTEXT_NUM_DEVICES                      0x1083"""

lines = info_names.splitlines()

enum = """#[derive(Debug, Clone)]
#[repr(u32)]
pub enum ContextInfoNames {
"""

impl = """impl ContextInfoNames {
    pub fn as_cl_str(&self) -> &str {
        match self {
"""

try_from = """impl TryFrom<cl_uint> for ContextInfoNames {
    type Error = ();
    fn try_from(v: cl_uint) -> Result<Self, Self::Error> {
        match v {
"""

for line in lines:
    parts = line.split(" ")
    enum_name = parts[1].replace("CL_CONTEXT_", "")
    enum_name = ''.join(x.title() for x in enum_name.split("_"))
    enum = enum + "    " + enum_name + " = " + parts[-1] + ",\n"
    impl = impl + "            ContextInfoNames::" + enum_name + " => \"" + parts[1] + "\",\n"
    try_from = try_from + "            x if x == ContextInfoNames::" + enum_name + " as cl_uint => Ok(ContextInfoNames::" + enum_name + "),\n"

enum = enum + "}\n"
impl = impl + "        }\n    }\n}\n"
try_from = try_from + "            _ => Err(()),\n"
try_from = try_from + "        }\n    }\n}\n"

print(enum)
print(impl)
print(try_from)
