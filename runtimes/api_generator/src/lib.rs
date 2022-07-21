use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

#[proc_macro_attribute]
pub fn cl_get_platform_ids(_attrs: TokenStream, input: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(input as ItemFn);

    let name = input_fn.sig.ident.clone();

    let gen = quote! {
        #input_fn

        #[no_mangle]
        pub extern fn clGetPlatformIDs(
            num_entries: cl_uint,
            platforms_raw: *mut *mut c_void,
            num_platforms_raw: *mut cl_uint,
        ) -> cl_int {
            let num_platforms = unsafe { num_platforms_raw.as_ref() };
            let platforms = unsafe { platforms_raw.as_ref() };

            lcl_contract!(
                num_entries != 0 || !num_platforms.is_none(),
                "either num_platforms is not NULL or num_entries is not 0",
                CL_INVALID_VALUE
            );

            lcl_contract!(
                !platforms.is_none() || !num_platforms.is_none(),
                "num_platforms and platforms can not be NULL at the same time",
                CL_INVALID_VALUE
            );
            let all_platforms = #name();

            if !platforms.is_none() {
                let platforms_array = unsafe {
                    std::slice::from_raw_parts_mut(
                        platforms_raw as *mut *mut dyn framework::Platform,
                        num_entries as usize,
                    )
                };
                for i in 0..num_entries {
                    platforms_array[i as usize] = unsafe { *all_platforms[i as usize] };
                }
            }

            if !num_platforms.is_none() {
                unsafe {
                    *num_platforms_raw = all_platforms.len() as u32;
                };
            }
            return CL_SUCCESS;
        }
    };

    return gen.into();
}

#[proc_macro_attribute]
pub fn cl_get_device_ids(_attrs: TokenStream, input: TokenStream) -> TokenStream {
    let gen = quote! {
        #[no_mangle]
        pub extern "C" fn clGetDeviceIDs(
            platform: cl_platform_id,
            device_type: cl_device_type,
            num_entries: cl_uint,
            devices: *mut cl_device_id,
            num_devices: *mut cl_uint,
        ) -> cl_int {

        }
    };
    return input;
}
