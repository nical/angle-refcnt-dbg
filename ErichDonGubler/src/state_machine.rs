use std::fmt::{self, Debug, Formatter};

use crate::spanned::Spanned;

#[derive(Debug)]
pub(crate) struct CallStack {
    pub frames: Vec<StackFrame>,
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum StackFrame {
    ExternalCode { address: Address },
    Symbolicated(SymbolicatedStackFrame),
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct SymbolicatedStackFrame {
    pub module: String,
    pub symbol_name: String,
    // TODO: source line and column!
}
impl From<SymbolicatedStackFrame> for StackFrame {
    fn from(value: SymbolicatedStackFrame) -> Self {
        Self::Symbolicated(value)
    }
}

#[derive(Clone, Eq, PartialEq)]
pub(crate) struct Address {
    value: u64,
}

impl Address {
    pub fn new(value: u64) -> Self {
        Self { value }
    }
}

impl Debug for Address {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Self { value } = self;
        write!(f, "{value:#010X}")
    }
}

pub(crate) struct Event<'a> {
    pub id: u64,
    pub kind: Option<EventKind<'a>>,
    pub address: Spanned<Address>,
    pub count: Spanned<u64>,
    pub callstack: Spanned<CallStack>,
}

impl Event<'_> {
    pub fn kind_name(&self) -> EventKindName {
        self.kind.as_ref().into()
    }
}

pub(crate) enum EventKind<'a> {
    Start { variable_name: Spanned<String> },
    Modify(ModifyEvent<'a>),
}

pub(crate) struct ModifyEvent<'a> {
    pub rule_name: &'a str, // TODO: span on this!
    pub kind: ModifyEventKind,
}

pub(crate) enum ModifyEventKind {
    Increment,
    Decrement,
    Destructor,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum EventKindName {
    Start,
    Increment,
    Decrement,
    Destructor,
    Unidentified,
}

impl<'a> From<Option<&'a EventKind<'a>>> for EventKindName {
    fn from(value: Option<&'a EventKind>) -> Self {
        match value {
            None => Self::Unidentified,
            Some(kind) => match kind {
                EventKind::Start { variable_name } => Self::Start,
                EventKind::Modify(ModifyEvent { rule_name: _, kind }) => match kind {
                    ModifyEventKind::Increment => Self::Increment,
                    ModifyEventKind::Decrement => Self::Decrement,
                    ModifyEventKind::Destructor => Self::Destructor,
                },
            },
        }
    }
}

#[derive(Clone, Copy, Debug, knuffel::Decode, Eq, PartialEq)]
pub(crate) enum Classification {
    Increment,
    Decrement,
    Destructor(DestructorClassification),
}

impl From<Classification> for ModifyEventKind {
    fn from(value: Classification) -> Self {
        match value {
            Classification::Increment => Self::Increment,
            Classification::Decrement => Self::Decrement,
            Classification::Destructor { .. } => Self::Destructor,
        }
    }
}

#[derive(Clone, Copy, Debug, knuffel::Decode, Eq, PartialEq)]
pub(crate) struct DestructorClassification {
    #[knuffel(property)]
    set_value: u64,
}
