use std::{cmp::Ordering, collections::BTreeMap, fmt::Debug, mem};

use knuffel::{
    ast::{Literal, Node, SpannedNode, Value},
    decode::Context,
    errors::DecodeError,
    traits::ErrorSpan,
    Decode, DecodeScalar,
};

use crate::{
    matcher::{
        Balance, BalanceKind, Classify, Event, Events, ExecutionTreeNode, Local, StackFrame,
        TailFrames,
    },
    platform::Platform,
};

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct RefcountEventParsingConfig<P> {
    pub platform: P,
    pub classifiers: Vec<Classify>,
    pub balancers: Vec<Balance>,
}

impl<S, P> knuffel::DecodeChildren<S> for RefcountEventParsingConfig<P>
where
    P: Platform<S>,
    S: ErrorSpan,
{
    fn decode_children(
        children: &[SpannedNode<S>],
        ctx: &mut Context<S>,
    ) -> Result<Self, DecodeError<S>> {
        let (platform_specs, rules_specs) = children
            .into_iter()
            .partition::<Vec<_>, _>(|child| &**child.node_name == "platform");

        let platform = match platform_specs.len().cmp(&1) {
            Ordering::Equal => {
                let mut platforms_found = platform_specs
                    .into_iter()
                    .map(|node| P::from_kdl(node, ctx));
                let first_platform = platforms_found.next().unwrap()?;
                assert!(platforms_found.next().is_none());
                first_platform
            }
            Ordering::Less => {
                return Err(DecodeError::MissingNode {
                    message: "no `platform` was specified".to_owned(),
                })
            }
            Ordering::Greater => {
                let multi_platform_err = |spec: &SpannedNode<S>| {
                    DecodeError::unexpected(
                        spec,
                        "platform",
                        "multiple values of `platform` are not supported".to_owned(),
                    )
                };
                let mut specs = platform_specs.into_iter().skip(1);
                let last = specs.next_back().unwrap();
                for spec in specs {
                    ctx.emit_error(multi_platform_err(spec));
                }
                // TODO: It'd be really nice to use this last one to point at the first `platform`,
                // instead.
                return Err(multi_platform_err(last));
            }
        };

        let mut classifiers = Vec::new();
        let mut balancers = Vec::new();
        for node in rules_specs {
            match Rule::decode_node(&platform, node, ctx)? {
                Rule::Classify(classifier) => classifiers.push(classifier),
                Rule::Balance(matcher) => balancers.push(matcher),
            }
        }

        Ok(Self {
            platform,
            classifiers,
            balancers,
        })
    }
}

#[derive(Debug, Eq, PartialEq)]
enum Rule {
    Classify(Classify),
    Balance(Balance),
}

impl Rule {
    fn decode_node<P, S>(
        platform: &P,
        node: &SpannedNode<S>,
        ctx: &mut Context<S>,
    ) -> Result<Self, DecodeError<S>>
    where
        P: Platform<S>,
        S: ErrorSpan,
    {
        #[derive(knuffel::DecodeScalar)]
        enum RuleKind {
            Classify,
            Balance,
        }

        let kind = RuleKind::decode(
            &Value {
                type_name: node.type_name.clone(),
                literal: node.node_name.clone().map(Literal::String),
            },
            ctx,
        )?;
        let this = match kind {
            RuleKind::Classify => Self::Classify(Classify::decode_node(platform, node, ctx)?),
            RuleKind::Balance => Self::Balance(Balance::decode_node(platform, node, ctx)?),
        };
        Ok(this)
    }
}

impl Classify {
    fn decode_node<P, S>(
        platform: &P,
        node: &SpannedNode<S>,
        ctx: &mut Context<S>,
    ) -> Result<Self, DecodeError<S>>
    where
        P: Platform<S>,
        S: ErrorSpan,
    {
        #[derive(knuffel::Decode)]
        struct ClassifyInner {
            #[knuffel(property)]
            name: String,
        }

        let ClassifyInner { name } = {
            let inner = node.clone().map(|node| {
                let Node {
                    type_name,
                    node_name,
                    arguments,
                    properties,
                    children: _,
                } = node;
                Node {
                    type_name,
                    node_name,
                    arguments,
                    properties,
                    children: None,
                }
            });
            ClassifyInner::decode_node(&inner, ctx)?
        };

        let (classification, tail_frames_matcher) =
            if let Some(children) = &node.children {
                let mut exec_tree;
                let exec_tree_span;
                {
                    let Events(matchers) = Events::decode_children(platform, children, ctx)?;
                    match matchers.len() {
                        1 => (),
                        0 => return Err(DecodeError::Missing {
                            span: children.span().clone(),
                            message:
                                "expected a single `execution` matcher within a classifier, got no \
                                children elements"
                                    .to_owned(),
                        }),
                        count => {
                            return Err(DecodeError::Unexpected {
                                span: children.span().clone(),
                                kind: "event matcher kind",
                                message: format!(
                                "expected a single `execution` matcher within a classifier, got \
                                {count} children"
                            ),
                            })
                        }
                    };
                    let mut matchers = matchers.into_iter();
                    exec_tree_span = children.first().unwrap().span().clone();
                    exec_tree = match matchers.next().unwrap() {
                        Event::TailFrames(_) => {
                            return Err(DecodeError::Unexpected {
                                span: children.span().clone(),
                                kind: "event matcher kind",
                                message: format!(
                                "expected a single `execution` matcher within a classifier, got a \
                                `tail-frames` matcher instead"
                            ),
                            })
                        }
                        Event::Execution(mut exec_trees) => match exec_trees.len() {
                            1 => exec_trees.pop().unwrap(),
                            0 => return Err(DecodeError::Missing {
                                span: exec_tree_span,
                                message:
                                    "expected a single child inside this `execution` matcher, got \
                                    no children elements"
                                        .to_owned(),
                            }),
                            count => {
                                return Err(DecodeError::Unexpected {
                                    span: exec_tree_span,
                                    kind: "event matcher kind",
                                    message: format!(
                                        "expected a single child inside this `execution` matcher, \
                                        got {count} children"
                                    ),
                                })
                            }
                        },
                    };
                }

                let mut stack_frame_matchers = Vec::new();
                let classification = loop {
                    match exec_tree {
                        ExecutionTreeNode::Frame {
                            matcher,
                            mut children,
                        } => {
                            match children.len() {
                            0 => return Err(DecodeError::Missing {
                                span: exec_tree_span.clone(),
                                message:
                                    "expected either a single stack frame activation or refcount \
                                    operation, but got neither".to_owned()
                            }),
                            1 => (),
                            count => return Err(DecodeError::Unexpected {
                                span: exec_tree_span.clone(),
                                kind: "children",
                                message: format!("expected a single child, but got {count}"),
                            })
                        }
                            stack_frame_matchers.push(matcher);
                            exec_tree = children.pop().unwrap();
                        }
                        ExecutionTreeNode::RefcountModification(classification) => {
                            break classification
                        }
                    }
                };
                (classification, TailFrames(stack_frame_matchers))
            } else {
                return Err(DecodeError::Missing {
                    span: node.span().clone(),
                    message: "expected a single `execution` block".to_owned(),
                });
            };

        Ok(Self {
            name,
            classification,
            tail_frames_matcher,
        })
    }
}

impl Balance {
    fn decode_node<P, S>(
        platform: &P,
        node: &SpannedNode<S>,
        ctx: &mut Context<S>,
    ) -> Result<Self, DecodeError<S>>
    where
        P: Platform<S>,
        S: ErrorSpan,
    {
        #[derive(Debug, knuffel::Decode)]
        struct BalanceOuter {
            #[knuffel(property(name = "type"))]
            type_: BalanceOuterKind,
            #[knuffel(property)]
            name: String,
        }

        #[derive(Debug, knuffel::DecodeScalar)]
        enum BalanceOuterKind {
            Local,
            Pair,
        }

        let outer;
        let mut inner;
        let children;
        {
            let mut node = node.clone();

            children = mem::take(&mut node.children);

            let outer_properties = {
                let mut props = BTreeMap::new();
                let mut move_to_inner = |key: &str| props.extend(node.properties.remove_entry(key));
                move_to_inner("type");
                move_to_inner("name");
                props
            };

            inner = node.clone().map(|node| {
                let Node {
                    type_name,
                    node_name,
                    arguments,
                    properties: _,
                    children: _,
                } = node;
                Node {
                    type_name,
                    node_name,
                    arguments,
                    properties: node.properties,
                    children: None,
                }
            });

            node.properties = outer_properties;
            outer = node;
        }

        let BalanceOuter { type_, name } = Decode::decode_node(&outer, ctx)?;

        let kind = match type_ {
            BalanceOuterKind::Local => {
                #[derive(Debug, knuffel::Decode)]
                struct LocalInner {}
                let LocalInner {} = Decode::decode_node(&inner, ctx)?;

                let children = match children {
                    Some(children) => children,
                    None => {
                        return Err(DecodeError::Missing {
                            span: node.span().clone(),
                            message: "missing children".to_owned(), // TODO: better diag.?
                        });
                    }
                };
                Local::decode_children(platform, &children, ctx).map(BalanceKind::Local)
            }
            BalanceOuterKind::Pair => {
                inner.children = children;
                Pair::decode_node(&inner, ctx)
                    .map(Into::into)
                    .map(BalanceKind::Pair)
            }
        }?;

        Ok(Self { name, kind })
    }
}

impl Events {
    fn decode_children<S, P>(
        platform: &P,
        children: &[SpannedNode<S>],
        ctx: &mut Context<S>,
    ) -> Result<Self, DecodeError<S>>
    where
        P: Platform<S>,
        S: ErrorSpan,
    {
        children
            .into_iter()
            .map(|child| Event::decode_from_node(platform, child, ctx))
            .collect::<Result<_, _>>()
            .map(Self)
    }
}

impl Event {
    fn decode_from_node<S, P>(
        platform: &P,
        node: &SpannedNode<S>,
        ctx: &mut Context<S>,
    ) -> Result<Self, DecodeError<S>>
    where
        P: Platform<S>,
        S: ErrorSpan,
    {
        let parser: fn(_, _, _) -> _ = match node.node_name.as_ref() {
            "tail-frames" => |platform, children, ctx| {
                TailFrames::decode_children(platform, children, ctx).map(Self::TailFrames)
            },
            "execution" => |platform, children, ctx| {
                ExecutionTreeNode::decode_children(platform, children, ctx).map(Self::Execution)
            },
            other => {
                return Err(DecodeError::Unexpected {
                    span: node.node_name.span().clone(),
                    kind: "event matcher node",
                    message: format!("node with unrecognized name {other:?}"),
                })
            }
        };

        let Node {
            type_name,
            node_name: _,
            arguments,
            properties,
            children,
        } = &**node;
        assert!(type_name.is_none());
        assert!(arguments.is_empty());
        assert!(properties.is_empty());

        let children = match children {
            None => {
                return Err(DecodeError::Missing {
                    span: node.span().clone(),
                    message: "expected children, got nothing".to_owned(),
                })
            }
            Some(children) => children,
        };

        parser(platform, children, ctx)
    }
}

impl TailFrames {
    fn decode_children<S, P>(
        platform: &P,
        children: &[SpannedNode<S>],
        ctx: &mut Context<S>,
    ) -> Result<Self, DecodeError<S>>
    where
        P: Platform<S>,
        S: ErrorSpan,
    {
        // TODO: dedupe the "just a node containing children" into its own helper
        let matchers = children
            .into_iter()
            .map(|node| StackFrame::decode_node(platform, node, ctx))
            .collect::<Result<_, _>>()?;
        Ok(Self(matchers))
    }
}

impl StackFrame {
    fn decode_node<S, P>(
        platform: &P,
        node: &SpannedNode<S>,
        ctx: &mut Context<S>,
    ) -> Result<Self, DecodeError<S>>
    where
        P: Platform<S>,
        S: ErrorSpan,
    {
        // TODO: I don't see an API for noting why the platform parsing failed after trying to
        // parse in a platform-agnostic way here. Maybe I missed it? Anyway, diagnostics suck
        // here. :(
        platform
            .decode_stack_frame_matcher_from_kdl(node, ctx)
            .or_else(|_e| Decode::decode_node(node, ctx))
    }
}

impl Local {
    fn decode_children<S, P>(
        platform: &P,
        children: &[SpannedNode<S>],
        ctx: &mut Context<S>,
    ) -> Result<Self, DecodeError<S>>
    where
        P: Platform<S>,
        S: ErrorSpan,
    {
        Ok(Self(Events::decode_children(platform, children, ctx)?))
    }
}

impl ExecutionTreeNode {
    fn decode_children<S, P>(
        platform: &P,
        children: &[SpannedNode<S>],
        ctx: &mut Context<S>,
    ) -> Result<Vec<Self>, DecodeError<S>>
    where
        P: Platform<S>,
        S: ErrorSpan,
    {
        // TODO: Maybe `emit` with a placeholder error at the end?
        children
            .into_iter()
            .map(|node| {
                let childless_node = node.clone().map(|node| {
                    let Node {
                        type_name,
                        node_name,
                        arguments,
                        properties,
                        children: _,
                    } = node;
                    Node {
                        type_name,
                        node_name,
                        arguments,
                        properties,
                        children: None,
                    }
                });
                StackFrame::decode_node(platform, &childless_node, ctx)
                    .map(|matcher| match &node.children {
                        None => Err(DecodeError::Missing {
                            span: node.span().clone(),
                            message:
                                "found a stack frame matching node in an `execution` tree, but no \
                            children were specified"
                                    .to_owned(),
                        }),
                        Some(children) => {
                            let children =
                                ExecutionTreeNode::decode_children(platform, &children, ctx)?;
                            Ok(Self::Frame { matcher, children })
                        }
                    })
                    // TODO: Booo, these diagnostics are gonna suck without more effort!
                    .or_else(|_e| {
                        Ok(Ok(Self::RefcountModification(Decode::decode_node(
                            node, ctx,
                        )?)))
                    })
                    .and_then(|res| res)
            })
            .collect()
    }
}

#[derive(Debug, knuffel::Decode, Eq, PartialEq)]
struct Pair {
    #[knuffel(property)]
    pub increment: u64,
    #[knuffel(property)]
    pub decrement: u64,
}

impl From<Pair> for crate::matcher::Pair {
    fn from(value: Pair) -> Self {
        let Pair {
            increment,
            decrement,
        } = value;
        crate::matcher::Pair {
            increment,
            decrement,
        }
    }
}
