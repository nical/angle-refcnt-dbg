use antileak::platform::windows::Windows;
use chumsky::{error::SimpleReason, Parser};
use format::lazy_format;
use knuffel::{ast::SpannedNode, decode::Context, errors::DecodeError, traits::ErrorSpan, Decode};
use miette::miette;

use crate::matcher;

use super::{Platform, PlatformKdl};

impl<S> Platform<S> for Windows
where
    S: ErrorSpan,
{
    fn from_kdl(node: &SpannedNode<S>, ctx: &mut Context<S>) -> Result<Self, DecodeError<S>> {
        #[derive(knuffel::Decode)]
        struct WindowsPlatformNode {}

        let WindowsPlatformNode {} = WindowsPlatformNode::decode_node(node, ctx)?;

        Ok(Self)
    }
}

impl<S> PlatformKdl<S> for Windows
where
    S: ErrorSpan,
{
    fn decode_stack_frame_matcher_from_kdl(
        &self,
        node: &SpannedNode<S>,
        ctx: &mut Context<S>,
    ) -> Result<matcher::StackFrame, DecodeError<S>> {
        #[derive(Debug, knuffel::Decode)]
        struct JustAName {
            #[knuffel(node_name)]
            node_name: String,
        }
        let JustAName { node_name } = Decode::decode_node(node, ctx)?;
        let span = node.node_name.span();

        Self::symbolicated_stack_frame_parser()
            .parse(&*node_name)
            .map(Into::into)
            .map_err(|mut errs| {
                assert_eq!(errs.len(), 1);
                let e = errs.pop().unwrap(); // TODO: oof, what if there's more than one?

                // TODO: We should probably try using a custom error type here? :think:
                let expected = lazy_format!(|f| {
                    let mut expected = e.expected();
                    if let Some(expected) = expected.next() {
                        write!(f, "{expected:?}")?;
                    }
                    for expected in e.expected() {
                        write!(f, ", {expected:?}")?;
                    }
                    Ok(())
                });
                match e.reason().clone() {
                    SimpleReason::Unexpected => match e.found() {
                        Some(found) => DecodeError::Unexpected {
                            span: span.clone(),
                            kind: "symbolicated stack frame",
                            message: format!("expected one of {expected}, found {found:?}",),
                        },
                        None => DecodeError::Missing {
                            span: span.clone(),
                            message: format!("missing {expected}"),
                        },
                    },
                    SimpleReason::Unclosed { .. } => unreachable!(),
                    SimpleReason::Custom(source) => DecodeError::Conversion {
                        span: span.clone(),
                        source: miette!(source).into(),
                    },
                }
            })
    }
}
