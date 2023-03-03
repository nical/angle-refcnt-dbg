use std::fmt::Display;

use chumsky::{prelude::Simple, Parser};

use crate::state_machine::{self, Address, CallStack};

pub mod windows;

pub trait PlatformLogSyntax {
    fn address_parser(&self) -> Box<dyn Parser<char, Address, Error = Simple<char>>>;
    fn display_address(&self) -> Box<dyn Display + '_>;
    fn refcount_address_count_pair_parser(
        &self,
    ) -> Box<dyn Parser<char, (Address, u64), Error = Simple<char>>>;
    fn stack_frame_parser(
        &self,
    ) -> Box<dyn Parser<char, state_machine::StackFrame, Error = Simple<char>>>;
    fn display_stack_frame(&self, frame: &state_machine::StackFrame) -> Box<dyn Display + '_>;
    fn call_stack_parser(&self) -> Box<dyn Parser<char, CallStack, Error = Simple<char>>>;
}
