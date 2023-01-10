use std::str::FromStr;

use anyhow::anyhow;

use crate::{CallStack, RefcountModifyKind, StackFrame};

#[derive(Debug, knuffel::Decode, Eq, PartialEq)]
pub(crate) struct RefcountEventParsingConfig {
    // TODO: Get a better diagnostic upstream, this failed hard when I specified `children`
    // erroneously
    #[knuffel(child, unwrap(children))]
    pub stack_matchers: Vec<CallStackMatcher>,
}

#[derive(Debug, knuffel::Decode, Eq, PartialEq)]
pub(crate) struct CallStackMatcher {
    #[knuffel(node_name)]
    classification: Classification,
    #[knuffel(child, unwrap(children))]
    top: Vec<StackFrameMatcher>,
}

impl CallStackMatcher {
    pub(crate) fn matches(&self, callstack: &CallStack) -> Option<Classification> {
        let Self {
            classification,
            top,
        } = self;
        top.iter()
            .zip(callstack.frames.iter())
            .all(|(matcher, frame)| matcher.matches(frame))
            .then_some(*classification)
    }
}

#[derive(Debug, knuffel::Decode, Eq, PartialEq)]
pub(crate) enum StackFrameMatcher {
    ExternalCode,
    Symbolicated(SymbolicatedStackFrameMatcher),
}

#[derive(Debug, knuffel::Decode, Eq, PartialEq)]
pub(crate) struct SymbolicatedStackFrameMatcher {
    #[knuffel(property)]
    pub(crate) module: Option<String>,
    #[knuffel(property)]
    pub(crate) symbol_name: Option<String>,
}

impl StackFrameMatcher {
    pub fn matches(&self, callstack: &StackFrame) -> bool {
        match (self, callstack) {
            (Self::ExternalCode, StackFrame::ExternalCode { .. }) => true,
            (
                Self::Symbolicated(SymbolicatedStackFrameMatcher {
                    module: m,
                    symbol_name: sn,
                }),
                StackFrame::Symbolicated {
                    module,
                    symbol_name,
                },
            ) => {
                m.as_ref().map_or(true, |m| m == module)
                    && sn.as_ref().map_or(true, |sn| sn == symbol_name)
            }
            _ => false,
        }
    }
}

#[derive(Clone, Copy, Debug, knuffel::DecodeScalar, Eq, PartialEq)]
pub(crate) enum Classification {
    Increment,
    Decrement,
    Destructor,
}

impl FromStr for Classification {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "increment" => Ok(Self::Increment),
            "decrement" => Ok(Self::Decrement),
            "destructor" => Ok(Self::Destructor),
            _ => Err(anyhow!("{s:?} is not a valid call stack classification")),
        }
    }
}

impl From<Classification> for RefcountModifyKind {
    fn from(value: Classification) -> Self {
        match value {
            Classification::Increment => Self::Increment,
            Classification::Decrement => Self::Decrement,
            Classification::Destructor => Self::Destructor,
        }
    }
}
