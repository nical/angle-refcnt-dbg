use std::fmt::Display;

use chumsky::{prelude::Simple, Parser};
use knuffel::{
    ast::{Node, SpannedNode},
    decode::Context,
    errors::{DecodeError, ExpectedType},
    traits::ErrorSpan,
    DecodeScalar,
};

use crate::{
    matcher,
    platform::windows::Windows,
    state_machine::{self, Address, CallStack},
};

mod windows;

pub(crate) trait Platform<S>
where
    Self: PlatformKdl<S> + PlatformLogSyntax + Sized,
    S: ErrorSpan,
{
    fn from_kdl(node: &SpannedNode<S>, ctx: &mut Context<S>) -> Result<Self, DecodeError<S>>
    where
        S: ErrorSpan;
}

pub(crate) trait PlatformKdl<S>
where
    S: ErrorSpan,
{
    fn decode_stack_frame_matcher_from_kdl(
        &self,
        node: &SpannedNode<S>,
        ctx: &mut Context<S>,
    ) -> Result<matcher::StackFrame, DecodeError<S>>
    where
        S: ErrorSpan;
}

pub(crate) trait PlatformLogSyntax {
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

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum SupportedPlatform {
    Windows(Windows),
}

impl SupportedPlatform {
    fn kdl<S>(&self) -> &dyn PlatformKdl<S>
    where
        S: ErrorSpan,
    {
        match self {
            Self::Windows(w) => w,
        }
    }

    fn log_syntax(&self) -> &dyn PlatformLogSyntax {
        match self {
            Self::Windows(w) => w,
        }
    }
}

impl<S> Platform<S> for SupportedPlatform
where
    S: ErrorSpan,
{
    fn from_kdl(node: &SpannedNode<S>, ctx: &mut Context<S>) -> Result<Self, DecodeError<S>>
    where
        S: ErrorSpan,
    {
        let node = node.clone();
        let Node {
            type_name,
            node_name: _,
            arguments,
            properties,
            children,
        } = &*node;
        if let Some(type_name) = type_name {
            return Err(DecodeError::TypeName {
                span: type_name.span().clone(),
                found: Some((&**type_name).clone()),
                expected: ExpectedType::no_type(),
                rust_type: "SupportedPlatform",
            });
        }
        let argument = node.arguments.first().ok_or_else(|| DecodeError::Missing {
            span: node.span().clone(),
            message: "a single argument containing the platform from which analyzable artifacts \
                will come"
                .to_string(),
        })?;
        assert_eq!(arguments.len(), 1);
        assert!(properties.is_empty());
        assert!(children.is_none());
        let this = {
            #[derive(knuffel::DecodeScalar)]
            enum Discriminant {
                Windows,
            }
            let ctor = match Discriminant::decode(argument, ctx)? {
                Discriminant::Windows => {
                    &|node, ctx| Windows::from_kdl(node, ctx).map(Self::Windows)
                }
            };

            let node = node.clone().map(|node| {
                let Node {
                    type_name: _,
                    node_name,
                    mut arguments,
                    properties,
                    children,
                } = node;
                arguments.remove(0);
                Node {
                    type_name: None,
                    node_name,
                    arguments,
                    properties,
                    children,
                }
            });
            ctor(&node, ctx)?
        };
        Ok(this)
    }
}

impl<S> PlatformKdl<S> for SupportedPlatform
where
    S: ErrorSpan,
{
    fn decode_stack_frame_matcher_from_kdl(
        &self,
        node: &SpannedNode<S>,
        ctx: &mut Context<S>,
    ) -> Result<matcher::StackFrame, DecodeError<S>>
    where
        S: ErrorSpan,
    {
        self.kdl().decode_stack_frame_matcher_from_kdl(node, ctx)
    }
}

impl PlatformLogSyntax for SupportedPlatform {
    fn stack_frame_parser(
        &self,
    ) -> Box<dyn Parser<char, state_machine::StackFrame, Error = Simple<char>>> {
        self.log_syntax().stack_frame_parser()
    }

    fn display_stack_frame(&self, frame: &state_machine::StackFrame) -> Box<dyn Display + '_> {
        self.log_syntax().display_stack_frame(frame)
    }

    fn address_parser(&self) -> Box<dyn Parser<char, Address, Error = Simple<char>>> {
        self.log_syntax().address_parser()
    }

    fn display_address(&self) -> Box<dyn Display + '_> {
        self.log_syntax().display_address()
    }

    fn refcount_address_count_pair_parser(
        &self,
    ) -> Box<dyn Parser<char, (Address, u64), Error = Simple<char>>> {
        self.log_syntax().refcount_address_count_pair_parser()
    }

    fn call_stack_parser(&self) -> Box<dyn Parser<char, CallStack, Error = Simple<char>>> {
        self.log_syntax().call_stack_parser()
    }
}
