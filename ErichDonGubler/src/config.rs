use serde::Deserialize;

use crate::{CallStack, RefcountModifyKind, StackFrame};

#[derive(Debug, Deserialize)]
pub(crate) struct RefcountEventParsingConfig {
    #[serde(rename = "stack_matcher")]
    pub stack_matchers: Vec<CallStackMatcher>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct CallStackMatcher {
    classification: Classification,
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

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub(crate) enum StackFrameMatcher {
    ExternalCode,
    Symbolicated {
        module: Option<String>,
        symbol_name: Option<String>,
    },
}

impl StackFrameMatcher {
    pub fn matches(&self, callstack: &StackFrame) -> bool {
        match (self, callstack) {
            (Self::ExternalCode, StackFrame::ExternalCode { .. }) => true,
            (
                Self::Symbolicated {
                    module: m,
                    symbol_name: sn,
                },
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

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Classification {
    Increment,
    Decrement,
    Destructor,
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
