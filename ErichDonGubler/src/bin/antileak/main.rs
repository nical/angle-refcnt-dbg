#![allow(unreachable_code)]

use std::{
    fmt::{self, Debug, Display},
    fs,
    num::NonZeroUsize,
    path::PathBuf,
};

use antileak::{
    event_log::{self, EventLog},
    spanned::Spanned,
    state_machine::{self, EventKindName},
};
use arcstr::ArcStr;
use ariadne::{Color, Fmt};
use chumsky::{error::SimpleReason, prelude::Simple, Parser};
use clap::Parser as _;
use format::lazy_format;
use log::LevelFilter;
use miette::{
    bail, miette, Context, Diagnostic, IntoDiagnostic, LabeledSpan, MietteError,
    MietteSpanContents, Severity, SourceCode, SourceSpan, SpanContents,
};
use thiserror::Error;

use crate::{
    config::RefcountEventParsingConfig,
    matcher::{Balance, BalanceKind, Classify, Local, Pair},
    platform::SupportedPlatform,
};

mod config;
mod matcher;
mod platform;

#[derive(Debug, clap::Parser)]
struct CliArgs {
    #[clap(long, default_value = ".")]
    input_dir: PathBuf,
    #[clap(subcommand)]
    subcommand: CliSubcommand,
}

#[derive(Debug, clap::Parser)]
enum CliSubcommand {
    Lint,
    ShowExec,
}

macro_rules! config_basename {
    () => {
        "antileak.kdl"
    };
}

// OPT: intern frame names, source names?
#[derive(Debug, Eq, PartialEq)]
struct SourceLocation {
    file_path: PathBuf,
    line: u64,
}

fn main() -> miette::Result<()> {
    env_logger::builder()
        .filter_level(LevelFilter::Info)
        .parse_default_env()
        .init();

    let CliArgs {
        input_dir,
        subcommand,
    } = CliArgs::parse();

    let vs_output_window_text_path = input_dir.join("vs-output-window.txt");
    // TODO: better coercion to `ArcStr` plz
    let vs_output_window_text_path_str = vs_output_window_text_path.to_str().unwrap();
    let vs_output_window_text_path_str = ArcStr::from(vs_output_window_text_path_str.to_owned());

    // OPT: buffered reading a chunk at a time for parsing plz
    let vs_output_window_text: ArcStr = {
        fs::read_to_string(&vs_output_window_text_path)
            .into_diagnostic()
            .wrap_err_with(|| format!("failed to read {vs_output_window_text_path:?}"))?
            .into()
    };

    let RefcountEventParsingConfig {
        platform,
        classifiers,
        balancers,
    } = {
        let config_path = input_dir.join(config_basename!());
        knuffel::parse::<RefcountEventParsingConfig<SupportedPlatform>>(
            &config_path.to_str().unwrap(),
            &fs::read_to_string(&config_path)
                .into_diagnostic()
                .wrap_err_with(|| {
                    format!(
                        "failed to read configuration from {}",
                        config_path.display()
                    )
                })?,
        )
        .map_err(|source| {
            #[derive(Debug, Diagnostic, Error)]
            struct ConfigParseError {
                #[source]
                #[diagnostic_source]
                source: knuffel::Error,
            }

            impl Display for ConfigParseError {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    write!(
                        f,
                        concat!("failed to deserialize `", config_basename!(), "`")
                    )
                }
            }

            ConfigParseError { source }
        })?
    };

    log::info!("parsing refcount event log...");
    // OPT: We might be paying a minor perf penalty for not pre-allocating here?
    let (log_opt, errs) = EventLog::parser(platform).parse_recovery(&vs_output_window_text[..]);

    let mut reporter = ErrorBuilder {
        source: Source {
            vs_output_window_text_path_str,
            vs_output_window_text: vs_output_window_text.clone(),
        },
        highest_severity_seen: None,
    };

    if !errs.is_empty() {
        for e in errs {
            #[derive(Debug, Error)]
            struct LogParseError {
                inner: Simple<char>,
            }

            impl Display for LogParseError {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    let Self { inner, .. } = self;
                    write!(f, "failed to parse refcount event: ")?;
                    match inner.reason() {
                        SimpleReason::Unexpected => write!(f, "unexpected input"),
                        SimpleReason::Unclosed { span, delimiter } => {
                            todo!("use span {span:?} and delimiter {delimiter:?}")
                        }
                        SimpleReason::Custom(msg) => write!(f, "{msg}"),
                    }
                }
            }

            impl Diagnostic for LogParseError {
                fn labels(&self) -> Option<Box<dyn Iterator<Item = LabeledSpan> + '_>> {
                    let Self { inner } = self;
                    let label = if inner.expected().count() != 0 || inner.found().is_some() {
                        let found = inner.found();
                        Some(format!(
                            "expected {}, found {}",
                            lazy_format!(|f| { f.debug_list().entries(inner.expected()).finish() })
                                .fg(Color::Green),
                            lazy_format!("{found:?}").fg(Color::Magenta),
                        ))
                    } else {
                        None
                    };
                    let span = inner.span();
                    Some(Box::new(
                        [LabeledSpan::new(label, span.start, span.end - span.start)].into_iter(),
                    ))
                }

                fn help<'a>(&'a self) -> Option<Box<dyn Display + 'a>> {
                    let label = self.inner.label();
                    Some(Box::new(lazy_format!(
                        "The parser state was labelled as {}, in case it helps with debugging. Hit \
                        Erich up; he probably borked something.",
                        lazy_format!("{label:?}").fg(Color::Yellow)
                    )))
                }
            }
            reporter.report_diag(LogParseError { inner: e })
        }
        bail!("one or more errors found during parsing; see above logs for more details");
    }

    let log = log_opt
        .ok_or_else(|| miette!("unrecoverable errors encountered while parsing, aborting"))?;

    log.events
        .iter()
        .enumerate()
        .skip(175)
        .take(10)
        .for_each(|(idx, Spanned(_evt, span))| {
            #[derive(Debug, Diagnostic, Error)]
            #[error("event {idx}")]
            #[diagnostic(severity(advice))]
            struct ShowEvent {
                idx: usize,
                span: SourceSpan,
            }
            reporter.report_diag(ShowEvent {
                idx,
                span: dbg!(span.clone()).into(),
            })
        });

    run(log, &classifiers, &balancers, reporter, subcommand)

    // TODO: Prompt ideas:
    // * Search for runs of balanced operations to triage for locally balanced stuff
    // * Order manual matching search by symbol prefix?
    // * For all prompt searches, show source, if available.

    // TODO: lint on source not being available, document how to get source lines?
}

// TODO: extract output channels as an interface, rather than using `{,e}print!`
fn run(
    event_log: EventLog,
    classifiers: &[Classify],
    balancers: &[Balance],
    mut reporter: ErrorBuilder,
    subcommand: CliSubcommand,
) -> miette::Result<()> {
    let EventLog { events } = event_log;

    // OPT: Would it be possible to operate on an arbitrary peek buffer and iterate by
    // original value instead, peeking when necessary? We're doing a lot of redundant
    // copying and classifying.
    let events = events
        .into_iter()
        .zip(0..)
        .map(|(parsed, id)| {
            let (event_log::Event { data, callstack }, evt_span) = parsed.into_parts();
            let event_log::EventData {
                address,
                count,
                kind,
            } = data;

            let classification = match kind {
                event_log::EventKind::Start { variable_name } => {
                    Some(state_machine::EventKind::Start { variable_name })
                }
                event_log::EventKind::Modify => classifiers
                    .iter()
                    .find_map(|c| c.matches(&callstack.as_inner().frames).map(|cl| cl.into()))
                    .map(|(rule_name, kind)| {
                        state_machine::EventKind::Modify(state_machine::ModifyEvent {
                            rule_name,
                            kind: kind.into(),
                        })
                    }),
            };

            Spanned::new(
                state_machine::Event {
                    id,
                    kind: classification,
                    address,
                    count,
                    callstack,
                },
                evt_span,
            )
        })
        .collect::<Vec<_>>();

    let mut events = events.iter();

    match subcommand {
        CliSubcommand::Lint => {
            fn start_event() -> impl Display {
                lazy_format!("{:?}", state_machine::EventKindName::Start).fg(Color::Green)
            }

            // TODO: handle multiple refcount lifecycles?
            let (start_event_span, count_at_start) = match events.next() {
                Some(Spanned(
                    evt @ state_machine::Event {
                        count: Spanned(count, count_span),
                        ..
                    },
                    evt_span,
                )) if evt.kind_name() == state_machine::EventKindName::Start => {
                    if *count == 0 {
                        #[derive(Debug, Diagnostic, Error)]
                        #[error("refcount start event has count of 0")]
                        struct BadStart {
                            #[label(
                                "this {} event has a refcount of 0, which is almost certainly \
                                wrong",
                                start_event()
                            )]
                            count_span: SourceSpan,
                        }
                        let err = BadStart {
                            count_span: count_span.clone().into(),
                        };
                        reporter.report_diag(err);
                    }
                    (evt_span, count)
                }
                None => {
                    log::info!("no events found, so log is trivially linted");
                    return Ok(());
                }
                Some(Spanned(event, span)) => {
                    #[derive(Debug, Diagnostic, Error)]
                    #[error("first refcount event is not a start event")]
                    #[diagnostic(help(
                        "This tool expects a single {start_event} event before all others. You \
                        should regenerate your log with a {start_event} event.",
                        start_event = start_event(),
                    ), severity(warning))]
                    struct ShouldaStartedFirst {
                        #[label(
                            "expected a {} event, but found this {} event instead",
                            start_event(),
                            lazy_format!("{event_kind_name:?}").fg(Color::Yellow),
                        )]
                        op_span: SourceSpan,
                        event_kind_name: EventKindName,
                    }
                    bail!(ShouldaStartedFirst {
                        op_span: span.clone().into(),
                        event_kind_name: event.kind_name(),
                    });
                }
            };

            let mut computed = ComputedRefcount {
                refcount: *count_at_start,
                num_unidentified: 0,
                num_dupe_start_events: 0,
            };

            fn misbehavior_help_msg() -> impl Display {
                lazy_format!(
                    "If you're looking for root causes of program misbehavior, then you may have \
                    found a lead!"
                )
            }

            fn destructor_event() -> impl Display {
                lazy_format!("{:?}", state_machine::EventKindName::Destructor).fg(Color::Magenta)
            }

            fn only_support_tracking_single_refcount_msg() -> impl Display {
                lazy_format!(
                    "This tool does not currently support more than a single lifecycle of single \
                    refcounted pointer. Make sure you record only a single refcount lifecycle for \
                    now."
                )
            }

            loop {
                // TODO: Unify diagnostic colors for event operation identifiers?
                // TODO: dedupe computed printing
                // OPT: We can probably used direct addition, instead of `checked_add`, since
                // we've already verified that everything can fit in memory.
                match events.next() {
                    // We should already have done something about this, so skippit.
                    Some(Spanned(event, span)) => {
                        let kind_name = event.kind_name();
                        let count = *event.count.as_inner();
                        match kind_name {
                            state_machine::EventKindName::Start => {
                                computed.num_dupe_start_events =
                                    computed.num_dupe_start_events.checked_add(1).unwrap();
                                #[derive(Debug, Diagnostic, Error)]
                                #[error("multiple {} events found", start_event())]
                                #[diagnostic(help(
                                    "{}",
                                    only_support_tracking_single_refcount_msg()
                                ))]
                                pub struct MultipleStartEvents {
                                    #[label("first event")]
                                    first: SourceSpan,
                                    #[label("second event for the same memory address")]
                                    second: SourceSpan,
                                }
                                reporter.report_sim_diag(
                                    computed.clone(),
                                    MultipleStartEvents {
                                        first: start_event_span.clone().into(),
                                        second: span.clone().into(),
                                    },
                                );
                            }
                            state_machine::EventKindName::Unidentified => {
                                computed.num_unidentified =
                                    computed.num_unidentified.checked_add(1).unwrap();

                                let suggested_match = match count {
                                    c if c == computed.refcount - 1 => Some("decrement"),
                                    c if c == computed.refcount.checked_add(1).unwrap() => {
                                        Some("increment")
                                    }
                                    _ => None,
                                };
                                #[derive(Debug, Diagnostic, Error)]
                                #[error(
                                    "failed to classify event with current configuration; \
                                    refcounts are going to be wrong!"
                                )]
                                #[diagnostic(help("{}", self.help()))]
                                struct Message {
                                    #[label]
                                    span: SourceSpan,
                                    suggested_match: Option<&'static str>,
                                }
                                impl Message {
                                    fn help(&self) -> impl Display + '_ {
                                        let Self {
                                            suggested_match, ..
                                        } = self;
                                        lazy_format!(move |f| {
                                            write!(
                                                f,
                                                concat!(
                                                    "You should change your stack matching \
                                                    configuration in your `",
                                                    config_basename!(),
                                                    "` so that it classifies this call stack \
                                                    correctly."
                                                ),
                                            )?;
                                            if let Some(suggested_match) = suggested_match {
                                                write!(
                                                    f,
                                                    " Perhaps this should be marked as {}?",
                                                    lazy_format!("{suggested_match:?}")
                                                        .fg(Color::Yellow)
                                                )?;
                                            }
                                            Ok(())
                                        })
                                    }
                                }
                                reporter.report_sim_diag(
                                    computed.clone(),
                                    Message {
                                        span: span.clone().into(),
                                        suggested_match,
                                    },
                                );
                            }
                            state_machine::EventKindName::Increment { .. } => {
                                computed.refcount = computed.refcount.checked_add(1).unwrap()
                                // TODO: `INFO`-level logging
                            }
                            state_machine::EventKindName::Decrement { .. } => {
                                computed.refcount = computed.refcount.checked_sub(1).unwrap();
                                // TODO: `INFO`-level logging
                                if computed.refcount == 0 {
                                    // TODO: `INFO`-level logging
                                    break;
                                }
                            }
                            state_machine::EventKindName::Destructor { .. } => {
                                #[derive(Debug, Diagnostic, Error)]
                                #[error("destructor called before refcount was zero")]
                                #[diagnostic(help("{}", misbehavior_help_msg()), severity(warning))]
                                struct DestructorTooEarly {
                                    #[label(
                                        "This {} operation happened while the refcount was \
                                        still positive, which is invalid.",
                                        destructor_event()
                                    )]
                                    op_span: SourceSpan,
                                    // TODO: track previous event span?
                                }
                                let err = DestructorTooEarly {
                                    op_span: span.clone().into(),
                                };
                                reporter.report_sim_diag(computed.clone(), err);
                                break;
                            }
                        };
                        if computed.refcount != count {
                            #[derive(Debug, Diagnostic, Error)]
                            #[error("refcount in log diverges from computed value")]
                            #[help(
                                "This is either a bug in this tool, or something _spooky_ in \
                                the logged code. {}",
                                misbehavior_help_msg()
                            )]
                            struct DivergentRefcount {
                                #[label(
                                    "This {} operation happened while the refcount was still \
                                    positive, which is invalid.",
                                    destructor_event()
                                )]
                                op_span: SourceSpan,
                                // TODO: track previous event span?
                            }
                            let err = DivergentRefcount {
                                op_span: span.clone().into(),
                            };
                            reporter.report_sim_diag(computed.clone(), err);
                        }
                    }
                    None => {
                        #[derive(Debug, Diagnostic, Error)]
                        #[error("log ends while refcount is above 0")]
                        #[diagnostic(help("{}", misbehavior_help_msg()), severity(warning))]
                        struct Leak {
                            #[label]
                            span: SourceSpan,
                        }

                        let err = Leak {
                            span: SourceSpan::new((reporter.source_len() - 1).into(), 1.into()),
                        };

                        reporter.report_sim_diag(computed.clone(), err);
                        break;
                    }
                }
            }

            fn warn_skipping_remaining(count: usize) {
                log::warn!("skipping processing of remaining {count} events");
            }
            fn skip_remaining(events: impl Iterator) {
                warn_skipping_remaining(events.count())
            }
            #[derive(Debug, Diagnostic, Error)]
            #[diagnostic(help("{}", misbehavior_help_msg()))]
            #[error("unexpected operations found after {}", kind.after_desc())]
            struct BadFinalOp {
                #[label(
                    "this {}{} operation and {remaining_event_count} other \
                    operation(s) were logged after {}",
                    match (kind, event_kind_name) {
                        (BadFinalOpKind::UseAfterFree, state_machine::EventKindName::Destructor)
                            => "extra",
                        _ => "",
                    },
                    lazy_format!("{:?}", event_kind_name).fg(Color::Red),
                    kind.after_desc(),
                )]
                offending_op: SourceSpan,
                event_kind_name: EventKindName,
                remaining_event_count: usize,
                kind: BadFinalOpKind,
                // TODO: point label at op right before this
            }

            #[derive(Debug)]
            enum BadFinalOpKind {
                UseAfterFree,
                ZeroButNotDestroyed,
            }

            impl BadFinalOpKind {
                pub fn after_desc(&self) -> &str {
                    match self {
                        BadFinalOpKind::UseAfterFree => "the refcount was destroyed",
                        BadFinalOpKind::ZeroButNotDestroyed => {
                            "the refcount reached 0 (but was not destroyed)"
                        }
                    }
                }
            }

            match events.clone().next() {
                Some(Spanned(
                    dtor_evt,
                    _, // TODO: use this span as part of a label
                )) if dtor_evt.kind_name() == state_machine::EventKindName::Destructor => {
                    if let Some(Spanned(event, span)) = events.nth(1) {
                        let remaining_event_count = events.count();
                        let err = BadFinalOp {
                            event_kind_name: event.kind_name(),
                            offending_op: dbg!(span.clone()).into(),
                            kind: BadFinalOpKind::UseAfterFree,
                            remaining_event_count,
                        };
                        reporter.report_sim_diag(computed.clone(), err);
                        warn_skipping_remaining(remaining_event_count);
                    }
                }
                Some(Spanned(event, span)) => {
                    let remaining_event_count = events.count();
                    let err = BadFinalOp {
                        event_kind_name: event.kind_name(),
                        offending_op: span.clone().into(),
                        kind: BadFinalOpKind::ZeroButNotDestroyed,
                        remaining_event_count,
                    };
                    reporter.report_sim_diag(computed.clone(), err);
                    warn_skipping_remaining(remaining_event_count);
                }
                None => {
                    // TODO: keep track of previous event's span, add this as a label and the
                    // offset of error message
                    #[derive(Debug, Diagnostic, Error)]
                    #[error(
                        "no events remain in log after the previous one (TODO: add span), but a {} \
                        event was expected",
                        destructor_event(),
                    )]
                    #[diagnostic(help("{}", misbehavior_help_msg()), severity(warning))]
                    struct NoDestructorAtEndError;
                    reporter.report_sim_diag(computed.clone(), NoDestructorAtEndError);
                }
            }

            if let Some(severity) = reporter.highest_severity_seen {
                log::log!(
                    severity_to_log_level(Some(severity)),
                    "found one or more issues while analyzing this log; see logs above for more \
                    details"
                );
            }
            Ok(())
        }
        CliSubcommand::ShowExec => {
            log::info!("parsing refcount events from log...");

            if events.clone().next().is_none() {
                println!("! no events found");
                return Ok(());
            }

            let mut prev_frames: &[_] = &[];
            loop {
                if events.clone().next().is_none() {
                    break;
                }

                let skipped = {
                    balancers.iter().find_map(|balance| {
                        let Balance { name, kind } = balance;
                        match kind {
                            BalanceKind::Local(Local(events_matcher)) => {
                                impl matcher::Events {
                                    pub(crate) fn matches<'a, I>(
                                        &self,
                                        iter: &mut I,
                                    ) -> Option<NonZeroUsize>
                                    where
                                        I: Clone + Iterator<Item = &'a state_machine::Event<'a>>,
                                    {
                                        let Self(matchers) = self;
                                        let mut matchers = matchers.iter().map(|m| m.matches(iter));
                                        matchers
                                            .next()
                                            .flatten()
                                            .map(|consumed| {
                                                matchers.try_fold(consumed, |acc, consumed| {
                                                    consumed.map(|consumed| {
                                                        acc.checked_add(consumed.get()).unwrap()
                                                    })
                                                })
                                            })
                                            .flatten()
                                    }
                                }
                                impl matcher::Event {
                                    pub(crate) fn matches<'a, I>(
                                        &self,
                                        iter: &mut I,
                                    ) -> Option<NonZeroUsize>
                                    where
                                        I: Clone + Iterator<Item = &'a state_machine::Event<'a>>,
                                    {
                                        match self {
                                            Self::TailFrames(_) => todo!(),
                                            Self::Execution(_) => todo!(),
                                        }
                                    }
                                }
                                events_matcher
                                    .matches(&mut events.clone().map(|Spanned(evt, _span)| evt))
                                    .map(|skipped| (skipped, name))
                            }
                            BalanceKind::Pair(Pair {
                                increment,
                                decrement,
                            }) => events
                                .clone()
                                .next()
                                .map(|Spanned(state_machine::Event { id, .. }, _span)| {
                                    [*increment, *decrement]
                                        .contains(&id)
                                        .then(|| (NonZeroUsize::new(1).unwrap(), name))
                                })
                                .flatten(),
                        }
                    })
                };

                let wind_up_frames;
                let op_display: &dyn Display;
                let skipped_display;
                let just_next_display;
                if let Some((skip_count, pattern_name)) = skipped {
                    let (frames, ids) = events.by_ref().take(skip_count.get()).fold(
                        (None, Vec::new()),
                        |(trimmed_frames, mut ids_skipped), evt| {
                            let state_machine::Event { id, callstack, .. } = evt.as_inner();
                            ids_skipped.push(id);
                            let these_frames = &callstack.as_inner().frames;
                            let trimmed_frames = trimmed_frames
                                .map(|trimmed_frames| {
                                    &these_frames[..common_frames_idx(trimmed_frames, these_frames)]
                                })
                                .unwrap_or(&these_frames[..]);
                            (Some(trimmed_frames), ids_skipped)
                        },
                    );
                    let frames = frames.unwrap();
                    events.nth(skip_count.get().checked_sub(1).unwrap()); // Skip stuff!
                    wind_up_frames = frames;
                    skipped_display = lazy_format!(move |f| {
                        write!(
                            f,
                            ": Skipping {skip_count} events (#{ids:?}) under balanced tree rule {pattern_name:?}",
                        )
                    });
                    // TODO: BUG: no unwind underscores being printed before skipped nodes
                    op_display = &skipped_display;
                } else {
                    let first_next = match events.next() {
                        Some(evt) => evt,
                        None => break,
                    };
                    let kind_name = first_next.as_inner().kind_name();
                    let (
                        state_machine::Event {
                            id,
                            kind: _,
                            count,
                            address,
                            callstack: Spanned(state_machine::CallStack { frames }, _),
                        },
                        _span,
                    ) = first_next.as_parts();
                    let count = count.as_inner();
                    wind_up_frames = frames;
                    just_next_display =
                        lazy_format!("* #{id}: {kind_name:?} of {address:?} -> {count}");
                    op_display = &just_next_display;
                }
                let wind_up_offset = wind_down(prev_frames, wind_up_frames);

                // Print wound-up frames.
                for (idx, frame) in wind_up_frames.iter().rev().enumerate().skip(wind_up_offset) {
                    let stack_frame = lazy_format!(|f| match frame {
                        state_machine::StackFrame::ExternalCode { address } =>
                            write!(f, "{address:?}"),
                        state_machine::StackFrame::Symbolicated(
                            state_machine::SymbolicatedStackFrame {
                                module,
                                symbol_name,
                            },
                        ) => write!(f, "{module}!{symbol_name}"),
                    });
                    println!("{:>frames_up$}\\ {stack_frame}", "", frames_up = idx);
                }
                println!("{:>frames_up$}|", "", frames_up = wind_up_frames.len());
                println!(
                    "{:>frames_up$}{op_display}",
                    "",
                    frames_up = wind_up_frames.len()
                );

                prev_frames = wind_up_frames;
            }
            wind_down(prev_frames, &[]);

            // TODO: `ShowExec` doesn't take multiple threads into account. Yikes!
            // TODO: Printing stats for matches would be nice!

            Ok(())
        }
    }
}

enum ExecutionEvent<'a> {
    PushStackFrame(&'a state_machine::StackFrame),
    RefcountOp(&'a state_machine::EventKind<'a>),
    PopStackFrame,
}

#[derive(Clone, Debug)]
struct ComputedRefcount {
    refcount: u64,
    num_unidentified: u64,
    num_dupe_start_events: u64,
}

struct ErrorBuilder {
    source: Source,
    highest_severity_seen: Option<Severity>,
}

impl ErrorBuilder {
    fn track<T>(&mut self, t: &T)
    where
        T: Debug + Diagnostic + Display,
    {
        #[derive(Eq, Ord, PartialEq, PartialOrd)]
        enum SeverityOrd {
            Error,
            Warning,
            Advice,
        }

        impl From<Severity> for SeverityOrd {
            fn from(value: Severity) -> Self {
                match value {
                    Severity::Error => SeverityOrd::Error,
                    Severity::Warning => SeverityOrd::Warning,
                    Severity::Advice => SeverityOrd::Advice,
                }
            }
        }

        impl From<SeverityOrd> for Severity {
            fn from(value: SeverityOrd) -> Self {
                match value {
                    SeverityOrd::Error => Severity::Error,
                    SeverityOrd::Warning => Severity::Warning,
                    SeverityOrd::Advice => Severity::Advice,
                }
            }
        }

        let severity = t.severity().unwrap_or(Severity::Error);
        self.highest_severity_seen =
            Some(self.highest_severity_seen.clone().map_or(severity, |s| {
                SeverityOrd::from(s).max(severity.into()).into()
            }));
    }

    fn report_diag<T>(&mut self, inner: T)
    where
        T: Debug + Diagnostic + Display + Send + Sync + 'static,
    {
        let full_err = LogAnalysisError {
            source_code: self.source.clone(),
            inner,
        };
        self.track(&full_err);
        let level = severity_to_log_level(full_err.severity());
        log::log!(level, "{:?}", miette::Report::new(full_err));
    }

    fn report_sim_diag<T>(&mut self, computed_refcount_state: ComputedRefcount, inner: T)
    where
        T: Debug + Diagnostic + Display + Send + Sync + 'static,
    {
        self.report_diag(SimError {
            computed_refcount_state: Computed(computed_refcount_state),
            inner,
        })
    }

    fn source_len(&self) -> usize {
        self.source.vs_output_window_text.len()
    }
}

#[derive(Clone, Debug)]
struct Source {
    vs_output_window_text_path_str: ArcStr,
    vs_output_window_text: ArcStr,
}

impl SourceCode for Source {
    fn read_span<'a>(
        &'a self,
        span: &SourceSpan,
        context_lines_before: usize,
        context_lines_after: usize,
    ) -> Result<Box<dyn SpanContents<'a> + 'a>, MietteError> {
        let Self {
            vs_output_window_text_path_str,
            vs_output_window_text,
        } = self;
        let contents =
            vs_output_window_text.read_span(span, context_lines_before, context_lines_after)?;
        Ok(Box::new(MietteSpanContents::new_named(
            (&**vs_output_window_text_path_str).to_owned(),
            contents.data(),
            *contents.span(),
            contents.line(),
            contents.column(),
            contents.line_count(),
        )))
    }
}

#[derive(Debug)]
struct LogAnalysisError<T>
where
    T: Debug + Diagnostic + Display,
{
    source_code: Source,
    inner: T,
}

impl<T> Diagnostic for LogAnalysisError<T>
where
    T: Debug + Diagnostic + Display,
{
    fn code<'a>(&'a self) -> Option<Box<dyn Display + 'a>> {
        None
    }

    fn severity(&self) -> Option<miette::Severity> {
        self.inner.severity()
    }

    fn help<'a>(&'a self) -> Option<Box<dyn Display + 'a>> {
        self.inner.help()
    }

    fn url<'a>(&'a self) -> Option<Box<dyn Display + 'a>> {
        self.inner.url()
    }

    fn source_code(&self) -> Option<&dyn SourceCode> {
        self.inner.source_code().or(Some(&self.source_code))
    }

    fn labels(&self) -> Option<Box<dyn Iterator<Item = miette::LabeledSpan> + '_>> {
        self.inner.labels()
    }

    fn related<'a>(&'a self) -> Option<Box<dyn Iterator<Item = &'a dyn Diagnostic> + 'a>> {
        self.inner.related()
    }

    fn diagnostic_source(&self) -> Option<&dyn Diagnostic> {
        self.inner.diagnostic_source()
    }
}

impl<T> Display for LogAnalysisError<T>
where
    T: Debug + Diagnostic + Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.inner, f)
    }
}

impl<T> std::error::Error for LogAnalysisError<T>
where
    T: Debug + Diagnostic + Display,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.inner.source()
    }
}

#[derive(Debug)]
struct SimError<T>
where
    T: Debug + Diagnostic + Display,
{
    computed_refcount_state: Computed,
    inner: T,
}

impl<T> Diagnostic for SimError<T>
where
    T: Debug + Diagnostic + Display,
{
    fn code<'a>(&'a self) -> Option<Box<dyn Display + 'a>> {
        None
    }

    fn severity(&self) -> Option<miette::Severity> {
        self.inner.severity()
    }

    fn help<'a>(&'a self) -> Option<Box<dyn Display + 'a>> {
        self.inner.help()
    }

    fn url<'a>(&'a self) -> Option<Box<dyn Display + 'a>> {
        self.inner.url()
    }

    fn source_code(&self) -> Option<&dyn SourceCode> {
        self.inner.source_code()
    }

    fn labels(&self) -> Option<Box<dyn Iterator<Item = miette::LabeledSpan> + '_>> {
        self.inner.labels()
    }

    fn related<'a>(&'a self) -> Option<Box<dyn Iterator<Item = &'a dyn Diagnostic> + 'a>> {
        let inner = self.inner.related().into_iter().flatten();
        let computed: Box<dyn Iterator<Item = &'a dyn Diagnostic>> = Box::new(
            Some(&self.computed_refcount_state)
                .into_iter()
                .map(|s| -> &dyn Diagnostic { &*s }),
        );
        Some(Box::new(inner.chain(computed)))
    }

    fn diagnostic_source(&self) -> Option<&dyn Diagnostic> {
        self.inner.diagnostic_source()
    }
}

impl<T> Display for SimError<T>
where
    T: Debug + Diagnostic + Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.inner, f)
    }
}

impl<T> std::error::Error for SimError<T>
where
    T: Debug + Diagnostic + Display,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.inner.source()
    }
}

#[derive(Debug, Diagnostic, Error)]
#[error("computed refcount state: {_0:#?}")]
#[diagnostic(severity(advice))]
struct Computed(ComputedRefcount);

fn severity_to_log_level(severity: Option<Severity>) -> log::Level {
    match severity.unwrap_or(Severity::Error) {
        Severity::Error => log::Level::Error,
        Severity::Warning => log::Level::Warn,
        Severity::Advice => log::Level::Info,
    }
}

fn wind_down(previous: &[state_machine::StackFrame], next: &[state_machine::StackFrame]) -> usize {
    let common_frames_idx = common_frames_idx(previous, next);
    println!(
        "{0:>space$}{0:_>frames_down$}|",
        "",
        space = common_frames_idx,
        frames_down = previous.len().checked_sub(common_frames_idx).unwrap()
    );
    common_frames_idx
}

fn common_frames_idx(
    previous: &[state_machine::StackFrame],
    next: &[state_machine::StackFrame],
) -> usize {
    if previous.is_empty() || next.is_empty() {
        return 0;
    }
    previous
        .iter()
        .rev()
        .zip(next.iter().rev())
        .position(|(f1, f2)| f1 != f2)
        .unwrap_or(previous.len())
}
