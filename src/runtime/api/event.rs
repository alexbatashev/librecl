use super::cl_types::*;
use crate::lcl_contract;
use crate::{
    api::error_handling::{map_invalid_event, ClError},
    interface::ContextImpl,
    interface::EventImpl,
    interface::EventKind,
};
use ocl_type_wrapper::cl_api;

#[cl_api]
fn clWaitForEvents(num_events: cl_uint, event_list: *const cl_event) -> Result<(), ClError> {
    lcl_contract!(
        num_events == 0,
        ClError::InvalidValue,
        "num_events must be greater than 0"
    );
    lcl_contract!(
        !event_list.is_null(),
        ClError::InvalidValue,
        "event_list can't be NULL"
    );

    let cl_events = unsafe { std::slice::from_raw_parts(event_list, num_events as usize) };

    let mut events = vec![];
    events.reserve(num_events as usize);
    for e in cl_events {
        events.push(EventKind::try_from_cl(*e).map_err(map_invalid_event)?);
    }

    let context = events[0]
        .get_context()
        .upgrade()
        .ok_or(ClError::InvalidEvent("failed to acquire owning reference to a context from the event. Was the context released before?".into()))?;

    context.wait_for_events(events.as_mut_slice())
}
