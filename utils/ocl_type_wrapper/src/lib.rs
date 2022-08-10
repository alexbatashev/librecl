extern crate proc_macro;
use std::ops::Deref;

use convert_case::{Case, Casing};
use darling::{FromDeriveInput, FromField};
use proc_macro::TokenStream;
use quote::format_ident;
use quote::quote;
use syn::{self, parse_macro_input, DeriveInput};

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
            ref_count: std::sync::atomic::AtomicUsize,
            handle: sync::SharedPtr<#dependent>,
        }

        impl #name {
            pub fn wrap(object: #dependent) -> *mut #name {
                let ptr = Box::into_raw(Box::new(#name {
                    dispatch_table: _cl_icd_dispatch::new(),
                    ref_count: std::sync::atomic::AtomicUsize::new(1),
                    handle: sync::SharedPtr::new(object),
                }));
                unsafe { ptr.as_mut().unwrap().handle.set_cl_handle(ptr) };

                return ptr;
            }

            pub fn retain(&mut self) {
                self.ref_count.fetch_add(1 as usize, std::sync::atomic::Ordering::SeqCst);
            }

            pub fn release(&mut self) -> usize {
                return self.ref_count.fetch_sub(1 as usize, std::sync::atomic::Ordering::SeqCst);
            }
        }

        impl FromCl<*mut #name> for #dependent {
            type Error = String;

            fn try_from_cl(value: *mut #name) -> Result<sync::SharedPtr<#dependent>, Self::Error> {
                match unsafe { value.as_ref() } {
                    Some(obj) => Ok(obj.handle.clone()),
                    None => Err("value is NULL".to_owned()),
                }
            }
        }
    };

    gen.into()
}

#[derive(Debug, FromField)]
#[darling(attributes(cl_handle), forward_attrs)]
struct HandleField {
    ident: Option<syn::Ident>,
    ty: syn::Type,
    #[allow(dead_code)]
    attrs: Vec<syn::Attribute>,
}

#[derive(Debug, FromDeriveInput)]
#[darling(attributes(cl_handle), supports(struct_named))]
struct ClObjImpl {
    #[allow(dead_code)]
    ident: syn::Ident,
    data: darling::ast::Data<darling::util::Ignored, HandleField>,
}

#[proc_macro_derive(ClObjImpl, attributes(cl_handle))]
pub fn derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input);
    let opts = ClObjImpl::from_derive_input(&input).expect("Wrong options");
    let DeriveInput { ident, .. } = input;

    // TODO need a better matching algorithm
    let fields = opts.data.take_struct().unwrap();
    let handle_field = fields.fields.last().as_ref().unwrap().clone();
    let cl_type_full = &handle_field.ty;
    let cl_type = match cl_type_full {
        syn::Type::Path(path) => match &path.path.segments[0].arguments {
            syn::PathArguments::AngleBracketed(ab) => match ab.args.first().as_ref().unwrap() {
                syn::GenericArgument::Type(gty) => match gty {
                    syn::Type::Path(p) => &p.path.segments[0].ident,
                    _ => panic!(),
                },
                _ => panic!(),
            },
            _ => &path.path.segments[0].ident,
        },
        _ => panic!("Not a path: {:?}", cl_type_full),
    };
    let field_name = handle_field.ident.as_ref().unwrap();

    let gen = quote! {
        impl ClObjectImpl<#cl_type> for #ident {
            fn get_cl_handle(&self) -> #cl_type {
                return *self.#field_name.value();
            }
            fn set_cl_handle(&mut self, handle: #cl_type) {
                self.#field_name = sync::UnsafeHandle::<#cl_type>::new(handle);
            }
        }
    };

    gen.into()
}

#[proc_macro_attribute]
pub fn cl_api(_args: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as syn::ItemFn);

    let body = input.block.clone();

    let func_name = input.sig.ident.clone();
    // TODO correctly handle IDs, KHR and LCL patterns.
    let func_name_str = func_name.clone().to_string();
    let func_name_processed = func_name_str
        .clone()
        .replace("IDs", "_ids")
        .replace("KHR", "_khr")
        .replace("LCL", "_lcl");
    let impl_name = format_ident!(
        "{}_impl",
        func_name_processed
            .from_case(Case::Camel)
            .to_case(Case::Snake)
    );

    let args = input.sig.inputs.clone();
    let arg_names = args.clone().into_iter().map(|arg| match arg {
        syn::FnArg::Typed(pat) => pat.pat,
        _ => panic!(),
    });

    let return_type = match input.sig.output {
        syn::ReturnType::Type(_, ret_type) => ret_type,
        _ => panic!(),
    };

    let cl_object_type = match return_type.deref() {
        syn::Type::Path(path) => {
            let first = path.path.segments.first();
            let result = first.unwrap();
            match result.arguments.clone() {
                syn::PathArguments::AngleBracketed(b) => {
                    let first = b.args.first().unwrap().clone();
                    match first {
                        syn::GenericArgument::Type(ty) => ty,
                        _ => panic!(),
                    }
                }
                _ => panic!(),
            }
        }
        _ => panic!(),
    };

    let api = match cl_object_type {
        syn::Type::Tuple(_) => {
            quote! {
                #[no_mangle]
                pub(crate) unsafe extern "C" fn #func_name(#args) -> cl_int {
                    let _span_ = tracing::span!(tracing::Level::TRACE, #func_name_str).entered();
                    let result = #impl_name(#(#arg_names),*);
                    match result {
                        Ok(_) => 0,
                        Err(err) => {
                            tracing::error!("{}", err);
                            err.error_code()
                        }
                    }
                }
            }
        }
        _ => {
            quote! {
                #[no_mangle]
                pub(crate) unsafe extern "C" fn #func_name(#args errorcode_ret: *mut cl_int) -> #cl_object_type {
                    let _span_ = tracing::span!(tracing::Level::TRACE, #func_name_str).entered();
                    let result = #impl_name(#(#arg_names),*);
                    match result {
                        Ok(ret) => {
                            if !errorcode_ret.is_null() {
                                *errorcode_ret = 0;
                            }
                            ret
                        },
                        Err(err) => {
                            tracing::error!("{}", err);
                            if !errorcode_ret.is_null() {
                                *errorcode_ret = err.error_code();
                            }
                            std::ptr::null_mut()
                        }
                    }
                }
            }
        }
    };

    let gen = quote! {
        fn #impl_name(#args) -> #return_type #body

        #api
    };

    gen.into()
}
