#![allow(unreachable_code)]

use std::{fmt::Display, fs, num::NonZeroUsize, ops::Range, path::PathBuf};

use arcstr::ArcStr;
use ariadne::{sources, Color, Fmt, Label, Report, ReportBuilder, ReportKind};
use chumsky::{error::SimpleReason, Parser};
use clap::Parser as _;
use format::lazy_format;
use log::LevelFilter;
use miette::{bail, ensure, miette, Context, IntoDiagnostic};

use crate::{
    config::RefcountEventParsingConfig,
    event_log::EventLog,
    matcher::{Balance, BalanceKind, Classify, Local, Pair},
    platform::SupportedPlatform,
    spanned::Spanned,
};

mod config;
mod event_log;
mod matcher;
mod platform;
mod spanned;
mod state_machine;

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
        .map_err(|e| {
            eprintln!("{:?}", miette::Report::new(e));

            miette!(concat!(
                "failed to deserialize `",
                config_basename!(),
                "`(see above for more details)"
            ))
        })?
    };

    log::info!("parsing refcount event log...");
    // OPT: We might be paying a minor perf penalty for not pre-allocating here?
    let (events_opt, errs) = EventLog::parser(platform).parse_recovery(&vs_output_window_text[..]);

    let reporter = Reporter {
        vs_output_window_text_path_str,
        vs_output_window_text: vs_output_window_text.clone(),
    };

    if !errs.is_empty() {
        // TODO: This is not a great way to determine if something is a warning or error.
        // We should let the error API of the parser report with finer granularity.
        let report_kind = if events_opt.is_some() {
            ReportKind::Warning // TODO: check if there are any of these cases left
        } else {
            ReportKind::Error
        };
        for e in errs {
            let msg = match e.reason() {
                SimpleReason::Unexpected => "unexpected input".to_owned(),
                SimpleReason::Unclosed { span, delimiter } => {
                    todo!("use span {span:?} and delimiter {delimiter:?}")
                }
                SimpleReason::Custom(msg) => msg.to_owned(),
            };

            let mut label = reporter.label(e.span()).with_color(Color::Red);
            if e.expected().count() != 0 || e.found().is_some() {
                let found = e.found();
                label = label.with_message(format!(
                    "Expected {}, found {}",
                    lazy_format!(|f| { f.debug_list().entries(e.expected()).finish() })
                        .fg(Color::Green),
                    lazy_format!("{found:?}").fg(Color::Magenta),
                ))
            }
            reporter.report(report_kind, e.span().start, |report| {
                report.set_message(format!("Failed to parse refcount event from file: {msg}"));
                report.add_label(label);
                if let Some(label) = e.label() {
                    report.set_note(format!(
                        "The parser state was labelled as {}, in case it helps with \
                                debugging. Hit Erich up; he probably borked something.",
                        lazy_format!("{label:?}").fg(Color::Yellow)
                    ));
                }
            });
        }
        bail!("one or more errors found during parsing; see above logs for more details");
    }

    let events = events_opt
        .ok_or_else(|| miette!("unrecoverable errors encountered while parsing, aborting"))?;

    run(events, &classifiers, &balancers, reporter, subcommand)

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
    reporter: Reporter,
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
            let start_event =
                lazy_format!("{:?}", state_machine::EventKindName::Start).fg(Color::Green);

            let mut found_issue = false;
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
                        reporter.report(ReportKind::Error, evt_span.start, |report| {
                            report.set_message("refcount start event has count of 0");
                            report.add_label(reporter.label(count_span.clone()).with_message(
                                format!(
                                "this {start_event} event has a refcount of 0, which is almost \
                                certainly wrong",
                            ),
                            ));
                        });
                        found_issue = true;
                    }
                    (evt_span, count)
                }
                None => {
                    log::info!("no events found, so log is trivially linted");
                    return Ok(());
                }
                Some(Spanned(event, span)) => {
                    reporter.report(ReportKind::Warning, span.start, |report| {
                        let start_event = lazy_format!("{:?}", state_machine::EventKindName::Start)
                            .fg(Color::Green);
                        report.set_message("first refcount event is not a start");
                        report.add_label(reporter.label(span.clone()).with_message(format!(
                            "expected a {start_event} event, but found this {} event instead",
                            lazy_format!("{:?}", event.kind_name()).fg(Color::Yellow),
                        )));
                        report.set_help(format!(
                            "This tool expects a single {start_event} event before all others. \
                            You should regenerate your log with a {start_event} event."
                        ));
                    });
                    // TODO: dedupe with below
                    bail!(
                        "found one or more issues while analyzing this log; see logs above for \
                        more details"
                    )
                }
            };

            let mut computed = ComputedRefcount {
                refcount: *count_at_start,
                num_unidentified: 0,
                num_dupe_start_events: 0,
            };
            let misbehavior_help_msg = "If you're looking for root causes of program misbehavior, \
                then you may have found a lead!";
            let destructor_event =
                lazy_format!("{:?}", state_machine::EventKindName::Destructor).fg(Color::Magenta);
            let only_support_one_thing_msg =
                "This tool does not currently support more than a single lifecycle of single \
                refcounted pointer. Make sure you record only a single refcount lifecycle for \
                now.";

            let computed_note = |computed: &_| format!("computed refcount state: {computed:?}");

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
                                reporter.report(ReportKind::Error, span.start, |report| {
                                    report.set_message("multiple start events found");
                                    report.set_help(only_support_one_thing_msg);
                                    report.add_label(reporter.label(span.clone()).with_message(
                                        "this {start_event} event occurs after the first one",
                                    ));
                                    report.add_label(
                                        reporter
                                            .label(start_event_span.clone())
                                            .with_message("this is the first event"),
                                    );
                                    report.set_note(computed_note(&computed));
                                });
                                found_issue = true;
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
                                reporter.report(ReportKind::Error, span.start, |report| {
                                    report.set_message("failed to classify event");
                                    report.add_label(reporter.label(span.clone()).with_message(
                                        "could not classify this event based on its call stack \
                                        from current configuration; refcounts are going to be \
                                        wrong!",
                                    ));
                                    // TODO: It might be nice to have a flag to accept and
                                    // persist automatic classifications!
                                    //
                                    // This might need to be a separate step, i.e., `antileak
                                    // analyze`.
                                    report.set_help(format!(
                                        concat!(
                                            "You should change your stack matching configuration \
                                            in your `",
                                            config_basename!(),
                                            "` so that it classifies this call stack \
                                            correctly.{}"
                                        ),
                                        lazy_format!(|f| {
                                            match suggested_match {
                                                None => Ok(()),
                                                Some(suggested_match) => write!(
                                                    f,
                                                    " Perhaps this should be marked as {}?",
                                                    lazy_format!("{suggested_match:?}")
                                                        .fg(Color::Yellow)
                                                ),
                                            }
                                        })
                                    ));
                                    report.set_note(computed_note(&computed));
                                });
                                found_issue = true;
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
                                reporter.report(ReportKind::Warning, span.start, |report| {
                                    report
                                        .set_message("destructor called before refcount was zero");
                                    // TODO: track previous event span?
                                    report.add_label(reporter.label(span.clone()).with_message(
                                        format!(
                                            "This {} operation happened while the refcount was \
                                            still positive, which is invalid.",
                                            destructor_event
                                        ),
                                    ));
                                    report.set_help(misbehavior_help_msg);
                                    report.set_note(computed_note(&computed));
                                });
                                found_issue = true;
                                break;
                            }
                        };
                        if computed.refcount != count {
                            reporter.report(ReportKind::Error, span.start, |report| {
                                report.set_message("refcount in log diverges from computed value");
                                // TODO: track previous event span?
                                report.add_label(reporter.label(span.clone()).with_message(
                                    format!(
                                        "This {} operation happened while the refcount was still \
                                positive, which is invalid.",
                                        destructor_event
                                    ),
                                ));
                                report.set_help(
                                    "This is either a bug in this tool, or something _spooky_ in \
                                    the logged code.",
                                );
                                report.set_note(computed_note(&computed));
                            });
                            found_issue = true;
                        }
                    }
                    None => {
                        reporter.report(
                            ReportKind::Warning,
                            reporter.source_len() - 1, // TODO: is this a sensible value?
                            |report| {
                                report.set_message("log ends while refcount is above 0");
                                report.set_note(computed_note(&computed));
                                report.set_help(misbehavior_help_msg);
                            },
                        );
                        found_issue = true;
                        break;
                    }
                }
            }

            match events.clone().next() {
                Some(Spanned(
                    dtor_evt,
                    _, // TODO: use this span as part of a label
                )) if dtor_evt.kind_name() == state_machine::EventKindName::Destructor => {
                    if let Some(Spanned(event, span)) = events.nth(1) {
                        let extra_maybe =
                            if event.kind_name() == state_machine::EventKindName::Destructor {
                                "extra "
                            } else {
                                ""
                            };
                        let remaining_event_count = events.count();
                        reporter.report(ReportKind::Error, span.start, |report| {
                            report.set_message("expected destructor call as final operation");
                            report.add_label(reporter.label(span.clone()).with_message(format!(
                                "this {extra_maybe}{} operation and {remaining_event_count} other \
                                refcount operation(s) were logged after the refcount reached 0 \
                                and was destroyed",
                                lazy_format!("{:?}", event.kind_name()).fg(Color::Red)
                            )));
                            report.set_note(format!("computed refcount state: {computed:?}"));
                            report.set_help(misbehavior_help_msg);
                        });
                        // TODO: point label at `Destructor` op right before this
                        found_issue = true;
                    }
                }
                Some(Spanned(event, span)) => {
                    reporter.report(ReportKind::Warning, span.start, |report| {
                        report.set_message("expected destructor call as final operation");
                        report.add_label(reporter.label(span.clone()).with_message(format!(
                            "this {} operation was called after the refcount was computed to \
                            reach 0, but a {destructor_event} event was expected",
                            lazy_format!("{:?}", event.kind_name()).fg(Color::Red)
                        )));
                        report.set_help(misbehavior_help_msg);
                        report.set_note(format!("computed refcount state: {computed:?}"));
                    });
                    found_issue = true;

                    let remaining_event_count = events.count();
                    log::warn!("skipping processing of remaining {remaining_event_count} events");
                }
                None => {
                    // TODO: keep track of previous event's span, add this as a label and the
                    // offset of error message
                    reporter.report(ReportKind::Warning, reporter.source_len() - 1, |report| {
                        report.set_message(format!(
                            "no events remain in log after the previous one (TODO: add span), \
                                but a {destructor_event} event was expected"
                        ));
                        // format!(
                        //     "expected destructor call as next operation after refcount was \
                        //         computed to reach 0, but no more events are found (computed: \
                        //         {computed:?})"
                        // ),
                        report.set_note(format!("computed refcount state: {computed:?}"));
                        report.set_help(misbehavior_help_msg);
                    });
                    found_issue = true;
                }
            }

            ensure!(
                !found_issue,
                "found one or more issues while analyzing this log; see logs above for more \
                details"
            );
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

struct Reporter {
    vs_output_window_text_path_str: ArcStr,
    vs_output_window_text: ArcStr,
}

impl Reporter {
    fn report(
        &self,
        kind: ReportKind,
        start: usize,
        configure: impl FnOnce(&mut ReportBuilder<(ArcStr, Range<usize>)>),
    ) {
        let Self {
            vs_output_window_text_path_str,
            vs_output_window_text,
        } = self;

        let mut report = Report::build(kind, vs_output_window_text_path_str.clone(), start);
        configure(&mut report);
        report
            .finish()
            .eprint(sources([(
                vs_output_window_text_path_str.clone(),
                vs_output_window_text.clone(),
            )]))
            .unwrap();
    }

    fn label(&self, span: Range<usize>) -> Label<(ArcStr, Range<usize>)> {
        Label::new((self.vs_output_window_text_path_str.clone(), span))
    }

    fn source_len(&self) -> usize {
        self.vs_output_window_text.len()
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
