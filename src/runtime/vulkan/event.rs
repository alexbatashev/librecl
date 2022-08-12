use crate::api::cl_types::*;
use crate::api::error_handling::ClError;
use crate::interface::ContextImpl;
use crate::interface::ContextKind;
use crate::interface::EventImpl;
use crate::interface::EventKind;
use crate::sync::{self, SharedPtr, UnsafeHandle, WeakPtr};
use ocl_type_wrapper::ClObjImpl;
use std::sync::Mutex;
use vulkano::command_buffer::CommandBufferExecFuture;
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use vulkano::sync::FenceSignalFuture;
use vulkano::sync::NowFuture;

// type Semaphore = SemaphoreSignalFuture<CommandBufferExecFuture<NowFuture, PrimaryAutoCommandBuffer>>;
type Fence = FenceSignalFuture<CommandBufferExecFuture<NowFuture, PrimaryAutoCommandBuffer>>;

#[derive(ClObjImpl)]
pub struct Event {
    _context: WeakPtr<ContextKind>,
    fence: SharedPtr<Fence>,
    #[cl_handle]
    handle: UnsafeHandle<cl_event>,
}

impl Event {
    pub fn new(context: WeakPtr<ContextKind>, fence: SharedPtr<Fence>) -> EventKind {
        Event {
            _context: context,
            fence,
            handle: UnsafeHandle::null(),
        }
        .into()
    }

    pub fn get_fence(&self) -> SharedPtr<Fence> {
        self.fence.clone()
    }
}

impl EventImpl for Event {}

#[derive(ClObjImpl)]
pub struct HostToGPUEvent {
    context: WeakPtr<ContextKind>,
    join_handle: SharedPtr<Mutex<Option<tokio::task::JoinHandle<()>>>>,
    #[cl_handle]
    handle: UnsafeHandle<cl_event>,
}

impl HostToGPUEvent {
    pub fn new(
        context: WeakPtr<ContextKind>,
        join_handle: tokio::task::JoinHandle<()>,
    ) -> EventKind {
        HostToGPUEvent {
            context,
            join_handle: SharedPtr::new(Mutex::new(Some(join_handle))),
            handle: UnsafeHandle::null(),
        }
        .into()
    }

    pub fn wait(&mut self) -> Result<(), ClError> {
        // TODO should we expect an error here?
        let context = self.context.upgrade().unwrap();
        // TODO handle poison error
        let mut handle_mutex = self.join_handle.lock().unwrap();
        if handle_mutex.is_some() {
            let join_handle = handle_mutex.take().unwrap();
            context
                .get_threading_runtime()
                .block_on(async move { join_handle.await })
                .map_err(|_| ClError::OutOfResources("failed to wait for host event".into()))?;
        }

        Ok(())
    }
}

impl EventImpl for HostToGPUEvent {}
