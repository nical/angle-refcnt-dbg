use crate::state_machine::{self, Classification};

/// A rule for classifying operations in a refcount log as increment, decrement, destructor, etc.
///
/// Classifiers consist of an event matching script containing a single refcount event. This event
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct Classify {
    pub name: String,
    pub tail_frames_matcher: TailFrames,
    pub classification: Classification,
}

impl Classify {
    pub(crate) fn matches(
        &self,
        callstack: &[state_machine::StackFrame],
    ) -> Option<(&str, Classification)> {
        let Self {
            tail_frames_matcher: tail_frames,
            name,
            classification,
        } = self;
        tail_frames
            .matches(callstack)
            .then_some((name, *classification))
    }
}
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct Balance {
    pub name: String,
    pub kind: BalanceKind,
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum BalanceKind {
    Local(Local),
    Pair(Pair),
}

/// A flavor of [`Matcher`] that matches locally balanced refcount modification events.
///
/// To be precise, this matcher contains a list of [`EventMatcher`]s that must emit only
/// [`RefcountOp`]
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct Local(pub Events);

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct Pair {
    pub increment: u64,
    pub decrement: u64,
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct Events(pub Vec<Event>);

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum Event {
    TailFrames(TailFrames),
    Execution(Vec<ExecutionTreeNode>),
}

#[derive(Debug, Default, Eq, PartialEq)]
pub(crate) struct TailFrames(pub(crate) Vec<StackFrame>);

impl TailFrames {
    pub fn matches(&self, callstack: &[state_machine::StackFrame]) -> bool {
        Self::matches_all([self], callstack)
    }

    pub fn matches_all<'a, I>(matchers: I, callstack: &[state_machine::StackFrame]) -> bool
    where
        I: IntoIterator<Item = &'a Self>,
        I::IntoIter: Clone + DoubleEndedIterator,
    {
        let matchers = matchers.into_iter().rev().flat_map(|matcher| {
            let Self(frame_matchers) = matcher;
            frame_matchers.iter().rev()
        });
        callstack.len() >= matchers.clone().count() && {
            callstack
                .iter()
                .zip(matchers)
                .all(|(frame, matcher)| matcher.matches(frame))
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum ExecutionTreeNode {
    Frame {
        matcher: StackFrame,
        children: Vec<ExecutionTreeNode>,
    },
    RefcountModification(Classification),
}

#[derive(Clone, Debug, knuffel::Decode, Eq, PartialEq)]
pub(crate) enum StackFrame {
    ExternalCode,
    Symbolicated(SymbolicatedStackFrame),
}

impl StackFrame {
    pub fn matches(&self, callstack: &state_machine::StackFrame) -> bool {
        match (self, callstack) {
            (Self::ExternalCode, state_machine::StackFrame::ExternalCode { .. }) => true,
            (
                Self::Symbolicated(SymbolicatedStackFrame {
                    module: m,
                    symbol_name: sn,
                }),
                state_machine::StackFrame::Symbolicated(state_machine::SymbolicatedStackFrame {
                    module,
                    symbol_name,
                }),
            ) => {
                m.as_ref().map_or(true, |m| m == module)
                    && sn.as_ref().map_or(true, |sn| sn == symbol_name)
            }
            _ => false,
        }
    }
}

impl From<SymbolicatedStackFrame> for StackFrame {
    fn from(value: SymbolicatedStackFrame) -> Self {
        Self::Symbolicated(value)
    }
}

impl From<state_machine::SymbolicatedStackFrame> for StackFrame {
    fn from(value: state_machine::SymbolicatedStackFrame) -> Self {
        SymbolicatedStackFrame::from(value).into()
    }
}

#[derive(Clone, Debug, knuffel::Decode, Eq, PartialEq)]
pub(crate) struct SymbolicatedStackFrame {
    #[knuffel(property)]
    pub module: Option<String>,
    #[knuffel(property)]
    pub symbol_name: Option<String>,
}

impl From<state_machine::SymbolicatedStackFrame> for SymbolicatedStackFrame {
    fn from(value: state_machine::SymbolicatedStackFrame) -> Self {
        let state_machine::SymbolicatedStackFrame {
            module,
            symbol_name,
        } = value;
        Self {
            module: Some(module),
            symbol_name: Some(symbol_name),
        }
    }
}
