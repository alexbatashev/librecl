use std::{
    ops::Deref,
    ops::DerefMut,
    sync::{Arc, Weak},
};

/// Analog to C++ std::shared_ptr
///
/// Unlike C++ std::shared_ptr, Rust std::sync::Arc does not allow taking mutable
/// references to contained data. This makes sense when the underlying object
/// is not thread-safe. OpenCL spec is written in a way, that any function may
/// be called from multiple threads, thus any cl_\* object *must* be thread-safe.
/// To overcome Arc's limitations, SharedPtr was introduced with a generic type,
/// that requires data object to be thread-safe.
pub struct SharedPtr<T: ?Sized + Sync + Send> {
    ptr: Arc<T>,
    phantom: std::marker::PhantomData<T>,
}

/// Analog of std::sync::Weak for SharedPtr.
pub struct WeakPtr<T: ?Sized + Sync + Send> {
    ptr: Weak<T>,
}

impl<T: Sized + Sync + Send> Clone for SharedPtr<T> {
    fn clone(&self) -> Self {
        return Self {
            ptr: self.ptr.clone(),
            phantom: self.phantom.clone(),
        };
    }
}

impl<T: Sized + Sync + Send> Clone for WeakPtr<T> {
    fn clone(&self) -> Self {
        return Self {
            ptr: self.ptr.clone(),
        };
    }
}

impl<T: Sized + Sync + Send> SharedPtr<T> {
    pub fn new(object: T) -> Self {
        SharedPtr {
            ptr: Arc::new(object),
            phantom: std::marker::PhantomData {},
        }
    }

    pub fn downgrade(ptr: &Self) -> WeakPtr<T> {
        let weak = Arc::downgrade(&ptr.ptr);
        return WeakPtr { ptr: weak };
    }
}

impl<T: ?Sized + Sync + Send> Deref for SharedPtr<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        return self.ptr.deref();
    }
}

impl<T: Sized + Send + Sync> DerefMut for SharedPtr<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        return unsafe { (Arc::as_ptr(&self.ptr) as *mut T).as_mut() }.unwrap();
    }
}

impl<T: Sized + Send + Sync> WeakPtr<T> {
    pub fn upgrade(&self) -> Option<SharedPtr<T>> {
        let shared = self.ptr.upgrade();
        match shared {
            Some(ptr) => Some(SharedPtr {
                ptr,
                phantom: std::marker::PhantomData,
            }),
            None => None,
        }
    }
}

/// Many OpenCL objects allow clSmthGetInfo calls, that return specific CL object
/// handles (like, parent context). This requires objects to somehow be aware of
/// their public handles. This simple wrapper provides a workaround to both store
/// the raw pointer and still comply with Sync + Send trait bounds.
///
/// `handle` should not be used outside of `api` crate.
///
/// # Examples
///
/// Most oftenly this thing will be used in the following context:
///
/// ```no_run
/// # use lcl_icd_runtime::api::cl_types::*;
/// # use lcl_icd_runtime::sync;
/// # use ocl_type_wrapper::*;
/// # use lcl_icd_runtime::sync::UnsafeHandle;
/// #[derive(ClObjImpl)]
/// struct Context {
///     #[cl_handle]
///     handle: UnsafeHandle<cl_context>,
/// }
///
/// impl Context {
///     pub fn new() -> Context {
///         Context {handle: UnsafeHandle::null()}
///     }
/// }
/// ```
///
/// When returning a newly created CL object, the `api` crate will wrap it by calling
/// _cl_smth::wrap(obj), which will automatically update `handle` field with
/// the public value.
///
/// There are two exceptions to this rule: platforms and devices must always
/// have the same value. In that case, one should manually call `::wrap(...)`
/// inside `new(...)` and return a shared pointer, obtained from raw handle
/// by calling `(Platform|Device)Kind::try_from_cl(raw).unwrap()`.
#[derive(Debug, Clone)]
pub struct UnsafeHandle<T> {
    ptr: Option<T>,
    phantom: std::marker::PhantomData<T>,
}

impl<T> UnsafeHandle<T> {
    pub fn new(ptr: T) -> Self {
        UnsafeHandle {
            ptr: Some(ptr),
            phantom: std::marker::PhantomData,
        }
    }

    pub fn null() -> Self {
        UnsafeHandle {
            ptr: None,
            phantom: std::marker::PhantomData,
        }
    }

    pub fn value(&self) -> &T {
        match &self.ptr {
            Some(ptr) => ptr,
            None => panic!("No value"),
        }
    }
}

unsafe impl<T> Sync for UnsafeHandle<T> {}
unsafe impl<T> Send for UnsafeHandle<T> {}

#[macro_export]
macro_rules! lcl_contract {
    ($cond:expr, $err_type:path, $message:expr) => {
        if !$cond {
            let message = $message;
            let func_name = stdext::function_name!();
            let assertion_message = format!(
                "Assertion {} failed in function {}: {}",
                stringify!($cond),
                func_name,
                message
            );
            let error = $err_type(assertion_message.into());
            return Err(error);
        }
    };
    ($context:tt, $cond:expr, $err_type:path, $message:expr) => {
        if !$cond {
            let message = $message;
            let func_name = stdext::function_name!();
            let assertion_message = format!(
                "Assertion {} failed in function {}: {}",
                stringify!($cond),
                func_name,
                message
            );
            let error = $err_type(assertion_message.into());
            $context.notify_error(format!("{}", error));
            return Err(error);
        }
    };
}

#[macro_export]
macro_rules! set_info_str {
    ($str:tt, $ptr:tt, $size_ptr:tt) => {{
        let ptr_safe = unsafe { $ptr.as_ref() };
        let size_ptr_safe = unsafe { $size_ptr.as_ref() };

        if size_ptr_safe.is_some() {
            unsafe { *$size_ptr = $str.len() as cl_size_t + 1 };
        }

        if ptr_safe.is_some() {
            unsafe {
                libc::strncpy(
                    $ptr as *mut libc::c_char,
                    $str.as_bytes().as_ptr() as *const libc::c_char,
                    $str.len() as libc::size_t,
                );
                *($ptr as *mut u8).offset($str.len() as isize) = 0;
            }
        }

        Result::<(), crate::api::error_handling::ClError>::Ok(())
    }};
}
#[macro_export]
macro_rules! set_info_int {
    ($ty:ty, $int:tt, $ptr:tt, $size_ptr:tt) => {{
        let ptr_safe = unsafe { $ptr.as_ref() };
        let size_ptr_safe = unsafe { $size_ptr.as_ref() };

        if !size_ptr_safe.is_none() {
            unsafe { *$size_ptr = std::mem::size_of::<$ty>() as cl_size_t };
        }

        if !ptr_safe.is_none() {
            unsafe {
                libc::memcpy(
                    $ptr as *mut libc::c_void,
                    &$int as *const $ty as *const libc::c_void,
                    std::mem::size_of::<$ty>() as libc::size_t,
                );
            }
        }

        Result::<(), crate::api::error_handling::ClError>::Ok(())
    }};
}
#[macro_export]
macro_rules! set_info_array {
    ($base_ty:ty, $array:tt, $ptr:tt, $size_ptr:tt) => {{
        let ptr_safe = unsafe { $ptr.as_ref() };
        let size_ptr_safe = unsafe { $size_ptr.as_ref() };

        if !size_ptr_safe.is_none() {
            unsafe { *$size_ptr = (std::mem::size_of::<$base_ty>() * $array.len()) as cl_size_t };
        }

        let dst = unsafe { std::slice::from_raw_parts_mut($ptr as *mut $base_ty, $array.len()) };

        if !ptr_safe.is_none() {
            for i in 0..$array.len() {
                dst[i] = $array[i].clone();
            }
        }

        Result::<(), crate::api::error_handling::ClError>::Ok(())
    }};
}
