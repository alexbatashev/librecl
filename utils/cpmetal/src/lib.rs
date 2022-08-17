mod other;

cfg_if::cfg_if! {
if #[cfg(target_os="macos")] {
mod darwin;
pub use crate::darwin::*;
} else {
pub use crate::other::*;
}
}
