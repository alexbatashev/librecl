extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;
use syn::{self, parse_macro_input};

#[proc_macro_attribute]
pub fn cl_object(args: TokenStream, item: TokenStream) -> TokenStream {
    let input_struct = parse_macro_input!(item as syn::ItemStruct);
    let args_parsed = parse_macro_input!(args as syn::Meta);

    let dependent = match args_parsed {
        syn::Meta::Path(path) => path,
        _ => panic!(),
    };

    let name = &input_struct.ident;

    let gen = quote! {
        #[repr(C)]
        pub struct #name {
            dispatch_table: *mut _cl_icd_dispatch,
            handle: std::rc::Rc<#dependent>,
        }

        impl #name {
            pub fn wrap(object: #dependent) -> *mut #name {
                return Box::into_raw(Box::new(#name {
                    dispatch_table: _cl_icd_dispatch::new(),
                    handle: std::rc::Rc::new(object),
                }));
            }
        }

        /*
        impl *mut #name {
            pub fn as_cl(&self) -> Rc<#dependent> {
                let unwrapped = unsafe { self.as_ref() }.unwrap();
                unwrapped.handle.clone()
            }
        }
        */
    };

    gen.into()
}
