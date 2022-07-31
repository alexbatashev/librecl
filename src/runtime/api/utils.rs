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
macro_rules! format_error {
    ($api_name_full:tt, $message:tt, $exit_code:tt) => {
        (|| -> String {
            let api_name = $api_name_full.split("::").last().unwrap_or("unknown");
            let bt = backtrace::Backtrace::new();
            let frame_id = (|bt: &backtrace::Backtrace| -> Option<usize> {
                for (id, f) in bt.frames().into_iter().enumerate().rev() {
                    let frame_id = f.symbols().into_iter().find_map(|s| {
                        if s.name()
                            .unwrap()
                            .as_str()
                            .unwrap_or("")
                            .starts_with(api_name)
                        {
                            return Some(id);
                        }
                        return Option::None;
                    });
                    if frame_id.is_some() {
                        return Some(frame_id.unwrap());
                    }
                }
                return Option::None;
            })(&bt);

            let mut symbol = Option::None;

            if frame_id.is_some() {
                symbol = Some(&bt.frames()[frame_id.unwrap() + 1].symbols()[0]);
            }

            if symbol.is_some() && symbol.unwrap().filename().is_some() {
                let final_message = std::format!(
                    "{}(...) -> {}: called from {} +{} ({}):\n{}\nBacktrace:\n{:?}\n",
                    api_name,
                    $exit_code,
                    symbol
                        .as_ref()
                        .unwrap()
                        .filename()
                        .unwrap_or(std::path::Path::new("unknown_file"))
                        .to_str()
                        .unwrap_or("unknown_file"),
                    symbol.as_ref().unwrap().lineno().unwrap_or(0),
                    symbol
                        .as_ref()
                        .unwrap()
                        .name()
                        .unwrap()
                        .as_str()
                        .unwrap_or("unknown function"),
                    $message,
                    bt
                );

                return final_message;
            }

            return std::format!(
                "{}(...) -> {}\n{}\nBacktrace:\n{:?}",
                api_name,
                $exit_code,
                $message,
                bt
            );
        })()
    };
}
#[macro_export]
macro_rules! lcl_contract {
    ($cond:expr, $message:tt, $exit_code:tt) => {
        if !$cond {
            let name = stdext::function_name!();
            let assertion_message =
                std::format!("Assertion {} failed: {}", stringify!($cond), $message);
            let full_message = format_error!(name, assertion_message, $exit_code);
            println!("{}", full_message);
            return $exit_code;
        }
    };
    ($cond:expr, $message:tt, $exit_code:tt, $ret_err:tt) => {
        if !$cond {
            let name = stdext::function_name!();
            let assertion_message =
                std::format!("Assertion {} failed: {}", stringify!($cond), $message);
            let full_message = format_error!(name, assertion_message, $exit_code);
            println!("{}", full_message);
            let ret_err_safe = unsafe { $ret_err.as_ref() };
            if ret_err_safe.is_some() {
                unsafe {
                    *$ret_err = $exit_code;
                }
            }
            return std::ptr::null_mut();
        }
    };
    ($ctx:tt, $cond:expr, $message:tt, $exit_code:tt) => {
        if !$cond {
            let name = stdext::function_name!();
            let assertion_message =
                std::format!("Assertion {} failed: {}", stringify!($cond), $message);
            let full_message = format_error!(name, assertion_message, $exit_code);
            $ctx.notify_error(full_message);
            return $exit_code;
        }
    };
    ($ctx:tt, $cond:expr, $message:tt, $exit_code:tt, $ret_err:tt) => {
        if !$cond {
            let name = stdext::function_name!();
            let assertion_message =
                std::format!("Assertion {} failed: {}", stringify!($cond), $message);
            let full_message = format_error!(name, assertion_message, $exit_code);
            $ctx.notify_error(full_message);
            let ret_err_safe = unsafe { $ret_err.as_ref() };
            if ret_err_safe.is_some() {
                unsafe {
                    *$ret_err = $exit_code;
                }
            }
            return std::ptr::null_mut();
        }
    };
}

#[macro_export]
macro_rules! set_info_str {
    ($str:tt, $ptr:tt, $size_ptr:tt) => {
        let ptr_safe = unsafe { $ptr.as_ref() };
        let size_ptr_safe = unsafe { $size_ptr.as_ref() };

        if !size_ptr_safe.is_none() {
            unsafe {
                *$size_ptr = $str.len() as cl_size_t + 1;
            }
        }

        if !ptr_safe.is_none() {
            unsafe {
                libc::strncpy(
                    $ptr as *mut libc::c_char,
                    $str.as_bytes().as_ptr() as *const libc::c_char,
                    $str.len() as libc::size_t,
                );
                *($ptr as *mut u8).offset($str.len() as isize) = 0;
            }
        }
    };
}
