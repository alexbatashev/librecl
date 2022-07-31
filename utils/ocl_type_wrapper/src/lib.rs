extern crate proc_macro;
use darling::{FromDeriveInput, FromField};
use proc_macro::TokenStream;
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
            handle: sync::SharedPtr<#dependent>,
        }

        impl #name {
            pub fn wrap(object: #dependent) -> *mut #name {
                let ptr = Box::into_raw(Box::new(#name {
                    dispatch_table: _cl_icd_dispatch::new(),
                    handle: sync::SharedPtr::new(object),
                }));
                unsafe { ptr.as_mut().unwrap().handle.set_cl_handle(ptr) };

                return ptr;
            }
        }

        impl FromCl<*mut #name> for #dependent {
            type Error = ();

            fn try_from_cl(value: *mut #name) -> Result<sync::SharedPtr<#dependent>, Self::Error> {
                match unsafe { value.as_ref() } {
                    Some(obj) => Ok(obj.handle.clone()),
                    None => Err(()),
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
    attrs: Vec<syn::Attribute>,
}

#[derive(Debug, FromDeriveInput)]
#[darling(attributes(cl_handle), supports(struct_named))]
struct ClObjImpl {
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

mod test {
    #[test]
    fn derive_macro_parsing() {}
}
