use std::sync::Mutex;

pub trait NativeCommandPool {
    type CommandBuffer: Send;

    fn create_command_buffers(&self, num_buffers: usize) -> Vec<Self::CommandBuffer>;
}

pub struct CommandPool<T: NativeCommandPool> {
    native_command_pool: Mutex<T>,
    // TODO this must be a thread local value, but I could not find anything
    // on crates.io
    ready_commands: Mutex<Box<Vec<T::CommandBuffer>>>,
}

impl<T: NativeCommandPool> CommandPool<T> {
    pub fn new(native_command_pool: T) -> Self {
        return CommandPool {
            native_command_pool: Mutex::new(native_command_pool),
            ready_commands: Mutex::new(Box::new(vec![])),
        };
    }

    pub fn get_command_buffer(&mut self) -> T::CommandBuffer {
        let mut commands = match self.ready_commands.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        if commands.is_empty() {
            let pool = match self.native_command_pool.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };

            commands
                .as_mut()
                .append(&mut pool.create_command_buffers(10 as usize));
        }

        return commands.pop().unwrap();
    }
}
