use chumsky::{
    prelude::Simple,
    primitive::{choice, just, take_until},
    text::ident,
    Parser,
};

use crate::{
    platform::PlatformLogSyntax,
    spanned::Spanned,
    state_machine::{Address, CallStack},
};

pub(crate) struct EventLog {
    pub events: Vec<Spanned<Event>>,
}

impl EventLog {
    pub fn parser<P>(platform: P) -> impl Parser<char, Self, Error = Simple<char>>
    where
        P: PlatformLogSyntax,
    {
        Event::parser(platform)
            .repeated()
            .at_least(1)
            .labelled("refcount event log")
            .map(|events| Self { events })
    }
}

macro_rules! start_of_event {
    () => {
        "--!! "
    };
}
const START_OF_EVENT: &str = start_of_event!();

#[derive(Debug)]
pub(crate) struct Event {
    pub data: EventData,
    pub callstack: Spanned<CallStack>,
}

impl Event {
    pub fn parser<S>(platform: S) -> impl Parser<char, Spanned<Self>, Error = Simple<char>>
    where
        S: PlatformLogSyntax,
    {
        take_until(just(START_OF_EVENT).labelled(concat!(
            "next refcount event sentinel (`",
            start_of_event!(),
            "`)"
        )))
        .ignore_then(
            choice((
                just("Ref count was changed for ")
                    .ignore_then(platform.refcount_address_count_pair_parser().map_with_span(
                        |(address, count), span| {
                            (
                                Spanned::new(address, span.clone()),
                                Spanned::new(count, span.clone()),
                            )
                        },
                    ))
                    .then_ignore(just(":"))
                    .then(
                        platform
                            .call_stack_parser()
                            .labelled("call stack")
                            .map_with_span(Spanned::new),
                    )
                    .map(|((address, count), callstack)| Self {
                        data: EventData {
                            address,
                            count,
                            kind: EventKind::Modify,
                        },
                        callstack,
                    }),
                just("Starting to track refs for `")
                    .ignore_then(
                        ident()
                            .labelled("refcounted data identifier")
                            .map_with_span(Spanned::new),
                    )
                    .then_ignore(just("` at "))
                    .then(platform.refcount_address_count_pair_parser().map_with_span(
                        |(address, count), span| {
                            (
                                Spanned::new(address, span.clone()),
                                Spanned::new(count, span.clone()),
                            )
                        },
                    ))
                    .then_ignore(just(":"))
                    .then(
                        platform
                            .call_stack_parser()
                            .labelled("call stack")
                            .map_with_span(Spanned::new),
                    )
                    .map(|((variable_name, (address, count)), callstack)| Self {
                        data: EventData {
                            address,
                            count,
                            kind: EventKind::Start { variable_name },
                        },
                        callstack,
                    }),
            ))
            .labelled("event body")
            .map_with_span(Spanned::new),
        )
        .labelled("refcount event")
    }
}

#[derive(Debug)]
pub(crate) struct EventData {
    pub address: Spanned<Address>,
    /// TODO: This doesn't take into account the size of the refcount maybe _not_ being 64 bits. We
    /// should do that, if this ever needs to be robust.
    pub count: Spanned<u64>,
    pub kind: EventKind,
}

#[derive(Debug)]
pub(crate) enum EventKind {
    Start { variable_name: Spanned<String> },
    Modify,
}
