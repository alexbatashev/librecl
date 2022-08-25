#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use lcl_derive::dyn_loader;

include!(concat!(env!("OUT_DIR"), "/compiler.rs"));

const LIB_NAME: &str = "liblcl_compiler.so";

dyn_loader!("OUT_DIR/compiler.rs", LIB_NAME, Context);
