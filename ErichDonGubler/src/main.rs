use std::{fs, num::NonZeroU64, ops::Range, path::PathBuf};

use anyhow::{anyhow, bail, Context};
use arcstr::ArcStr;
use ariadne::{sources, Color, Fmt, Label, Report, ReportBuilder, ReportKind};
use chumsky::{
    error::SimpleReason,
    prelude::Simple,
    primitive::{choice, end, filter, just, take_until},
    text::{ident, newline},
    Parser,
};
use clap::Parser as _;
use format::lazy_format;
use log::LevelFilter;
use serde::Deserialize;

#[derive(Debug, clap::Parser)]
struct CliArgs {
    #[clap(subcommand)]
    subcommand: CliSubcommand,
}

#[derive(Debug, clap::Parser)]
enum CliSubcommand {
    Run { input_dir: PathBuf },
}

macro_rules! start_of_event {
    () => {
        "--!! "
    };
}
const START_OF_EVENT: &str = start_of_event!();

macro_rules! config_toml {
    () => {
        "config.toml"
    };
}

#[derive(Debug, PartialEq)]
pub struct Spanned<T>(T, Range<usize>);

#[derive(Debug)]
enum RefcountEvent {
    // TODO: parse this
    Start {
        variable_name: String,
        address: Address,
        callstack: CallStack,
    },
    Increment {
        callstack: CallStack,
    },
    Decrement {
        callstack: CallStack,
    },
    Destructor {
        callstack: CallStack,
    },
    Unidentified {
        callstack: CallStack,
    },
}

#[derive(Clone, Debug)]
struct ComputedRefcount {
    refcount: u64,
    num_unidentified: u64,
    num_dupe_start_events: u64,
}

#[derive(Debug)]
enum RefcountEventKind {
    Start,
    Increment,
    Decrement,
    Destructor,
    Unidentified,
}

impl RefcountEvent {
    pub fn kind(&self) -> RefcountEventKind {
        match self {
            Self::Start { .. } => RefcountEventKind::Start,
            Self::Increment { .. } => RefcountEventKind::Increment,
            Self::Decrement { .. } => RefcountEventKind::Decrement,
            Self::Destructor { .. } => RefcountEventKind::Destructor,
            Self::Unidentified { .. } => RefcountEventKind::Unidentified,
        }
    }

    fn parser(
        stack_matchers: &[CallStackMatcher],
    ) -> impl Parser<char, Spanned<Self>, Error = Simple<char>> + '_ {
        take_until(just(START_OF_EVENT).labelled(concat!(
            "next refcount event sentinel (`",
            start_of_event!(),
            "`)"
        )))
        .ignore_then(
            choice((
                just("Ref count for device was modified:").ignore_then(
                    CallStack::parser().labelled("call stack").map(
                        |callstack| match stack_matchers
                            .iter()
                            .find_map(|matcher| matcher.matches(&callstack))
                        {
                            Some(classification) => match classification {
                                Classification::Increment => RefcountEvent::Increment { callstack },
                                Classification::Decrement => RefcountEvent::Decrement { callstack },
                                Classification::Destructor => {
                                    RefcountEvent::Destructor { callstack }
                                }
                            },
                            None => RefcountEvent::Unidentified { callstack },
                        },
                    ),
                ),
                just("Starting to track refs `")
                    .ignore_then(ident().labelled("COM object identifier"))
                    .then_ignore(just("` at "))
                    .then(just("0x").ignore_then(Address::parser().labelled("refcount address")))
                    .then_ignore(just(":"))
                    .then(CallStack::parser().labelled("call stack"))
                    .map(|((variable_name, address), callstack)| Self::Start {
                        variable_name,
                        address,
                        callstack,
                    }),
            ))
            .labelled("event body"),
        )
        .labelled("refcount event")
        .map_with_span(Spanned)
    }
}

#[derive(Debug)]
struct CallStack {
    frames: Vec<StackFrame>,
}

impl CallStack {
    fn parser() -> impl Parser<char, Self, Error = Simple<char>> {
        // OPT: We're probably paying a non-trivial amount here for parsing dozens of frames
        // without trying to set capacity first.
        StackFrame::parser()
            .repeated()
            .at_least(1)
            .then_ignore(just("\t").then(newline()))
            .labelled("`$CALLSTACK` output")
            .map(|frames| Self { frames })
    }
}

#[test]
fn parse_call_stack() {
    let data = "\td3d11.dll!TComObject<NOutermost::CDevice>::AddRef
\td3d11_3SDKLayers.dll!CLayeredObject<NDebug::CDevice>::CContainedObject::AddRef
\td3d11_3SDKLayers.dll!ATL::CComObjectRootBase::InternalQueryInterface
\td3d11_3SDKLayers.dll!CLayeredObject<NDebug::CDevice>::QueryInterface
\td3d11.dll!ATL::AtlInternalQueryInterface
\td3d11_3SDKLayers.dll!CLayeredObject<NDebug::CDevice>::CContainedObject::QueryInterface
\tlibGLESv2.dll!rx::Renderer11::initializeD3DDevice
\tlibGLESv2.dll!rx::Renderer11::initialize
\tlibGLESv2.dll!rx::CreateRendererD3D
\tlibGLESv2.dll!rx::DisplayD3D::initialize
\tlibGLESv2.dll!egl::Display::initialize
\tlibGLESv2.dll!egl::Initialize
\tlibGLESv2.dll!EGL_Initialize
\tlibEGL.dll!eglInitialize
\txul.dll!mozilla::gl::GLLibraryEGL::fInitialize
\txul.dll!mozilla::gl::EglDisplay::Create
\txul.dll!mozilla::gl::GetAndInitDisplay
\txul.dll!mozilla::gl::GetAndInitDisplayForAccelANGLE
\txul.dll!mozilla::gl::GLLibraryEGL::CreateDisplayLocked
\txul.dll!mozilla::gl::GLLibraryEGL::DefaultDisplay
\txul.dll!mozilla::gl::DefaultEglDisplay
\txul.dll!mozilla::gl::GLContextProviderEGL::CreateHeadless
\txul.dll!mozilla::WebGLContext::CreateAndInitGL::<lambda_0>::operator()
\txul.dll!mozilla::WebGLContext::CreateAndInitGL::<lambda>
\txul.dll!mozilla::WebGLContext::CreateAndInitGL
\txul.dll!mozilla::WebGLContext::Create::<lambda>
\txul.dll!mozilla::WebGLContext::Create
\txul.dll!mozilla::HostWebGLContext::Create
\txul.dll!mozilla::dom::WebGLParent::RecvInitialize
\txul.dll!mozilla::dom::PWebGLParent::OnMessageReceived
\txul.dll!mozilla::gfx::PCanvasManagerParent::OnMessageReceived
\txul.dll!mozilla::ipc::MessageChannel::DispatchSyncMessage
\txul.dll!mozilla::ipc::MessageChannel::DispatchMessage
\txul.dll!mozilla::ipc::MessageChannel::RunMessage
\txul.dll!mozilla::ipc::MessageChannel::MessageTask::Run
\txul.dll!nsThread::ProcessNextEvent
\txul.dll!NS_ProcessNextEvent
\txul.dll!mozilla::ipc::MessagePumpForNonMainThreads::Run
\txul.dll!MessageLoop::RunInternal
\txul.dll!MessageLoop::RunHandler
\txul.dll!MessageLoop::Run
\txul.dll!nsThread::ThreadFunc
\tnss3.dll!_PR_NativeRunThread
\tnss3.dll!pr_root
\tucrtbase.dll!thread_start<unsigned int (__cdecl*)(void *),1>
\t000002147c35002f
\tmozglue.dll!mozilla::interceptor::FuncHook<mozilla::interceptor::WindowsDllInterceptor<mozilla::interceptor::VMSharingPolicyShared>,void (*)(int, void *, void *)>::operator()<int &,void *&,void *&>
\tmozglue.dll!patched_BaseThreadInitThunk
\tntdll.dll!RtlUserThreadStart
\t\n";

    insta::assert_debug_snapshot!(CallStack::parser().parse(data).unwrap());
}

#[derive(custom_debug::Debug)]
#[cfg_attr(test, derive(Eq, PartialEq))]
enum StackFrame {
    ExternalCode {
        address: Address,
    },
    WithSymbols {
        module: String,
        symbol_name: String,
        source_location: Option<SourceLocation>,
    },
}

#[derive(custom_debug::Debug)]
#[debug(format = "{value:#010X}")]
#[cfg_attr(test, derive(Eq, PartialEq))]
struct Address {
    value: u64,
}

impl Address {
    fn parser() -> impl Parser<char, Self, Error = Simple<char>> {
        filter(char::is_ascii_hexdigit)
            .repeated()
            .exactly(16)
            .labelled("ASCII hex digits of addresss")
            // OPT: s/String/&str?
            .map(|digits| String::from_iter(digits.into_iter()))
            .map(|digits| u64::from_str_radix(&digits, 16).unwrap())
            .map(|value| Self { value })
    }
}

impl StackFrame {
    fn parser() -> impl Parser<char, Self, Error = Simple<char>> {
        fn non_newline(c: &char) -> bool {
            let mut buf = [0; 4];
            let s = &*c.encode_utf8(&mut buf);
            end::<Simple<char>>().parse(s).is_err() && newline::<Simple<char>>().parse(s).is_err()
        }
        let external_code = Address::parser()
            .map(|address| Self::ExternalCode { address })
            .labelled("external code");

        let chars_until = |pred: fn(&char) -> bool| {
            filter(pred)
                .repeated()
                .at_least(1)
                // TODO: max bounds?
                // OPT: eww, y u no `str`?
                .map(|cs| String::from_iter(cs))
        };
        let module = chars_until(|c| non_newline(c) && *c != '!')
            .then_ignore(just("!"))
            .labelled("module");
        let symbol_name = chars_until(non_newline).labelled("symbol name");
        let symbolicated_stack_frame =
            module
                .then(symbol_name)
                .map(|(module, symbol_name)| Self::WithSymbols {
                    module,
                    symbol_name,
                    source_location: None,
                });
        choice((external_code, symbolicated_stack_frame))
            .labelled("inner stack frame line")
            .delimited_by(just("\t"), newline())
            .labelled("full stack frame line")
    }
}

#[test]
fn parse_symbolicated_stack_frames() {
    assert_eq!(
        StackFrame::parser().parse("\ta!b\n").unwrap(),
        StackFrame::WithSymbols {
            module: "a".to_string(),
            symbol_name: "b".to_string(),
            source_location: None
        }
    );
    assert_eq!(
        StackFrame::parser().parse(
            "\td3d11_3SDKLayers.dll!CLayeredObject<NDebug::CDevice>::CContainedObject::Release\n"
        )
        .unwrap(),
        StackFrame::WithSymbols {
            module: "d3d11_3SDKLayers.dll".to_string(),
            symbol_name: "CLayeredObject<NDebug::CDevice>::CContainedObject::Release"
                .to_string(),
            source_location: None
        }
    );
    assert!(StackFrame::parser().parse("\td3d11_").is_err());
    assert!(StackFrame::parser().parse("\td3d11_\n").is_err());
    assert!(StackFrame::parser().parse("\td3d11!\n").is_err());
    assert!(StackFrame::parser().parse("\t!\n").is_err());
    assert!(StackFrame::parser().parse("\t!asdf\n").is_err());
}

// OPT: intern frame names, source names?
#[derive(Debug)]
#[cfg_attr(test, derive(Eq, PartialEq))]
struct SourceLocation {
    file_path: PathBuf,
    line: u64,
}

struct RefcountEventLog {
    events: Vec<Spanned<RefcountEvent>>,
}

#[derive(Debug, Deserialize)]
struct RefcountEventParsingConfig {
    /// TODO: This doesn't take into account the size of the refcount maybe _not_ being 64 bits. We
    /// should do that, if this ever needs to be robust.
    count_at_start: NonZeroU64,
    #[serde(rename = "stack_matcher")]
    stack_matchers: Vec<CallStackMatcher>,
}

#[derive(Debug, Deserialize)]
struct CallStackMatcher {
    classification: Classification,
    top: Vec<StackFrameMatcher>,
}

impl CallStackMatcher {
    pub fn matches(&self, callstack: &CallStack) -> Option<Classification> {
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
enum StackFrameMatcher {
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
                StackFrame::WithSymbols {
                    module,
                    symbol_name,
                    source_location: _,
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
enum Classification {
    Increment,
    Decrement,
    Destructor,
}

impl RefcountEventLog {
    fn parser(
        stack_matchers: &[CallStackMatcher],
    ) -> impl Parser<char, Self, Error = Simple<char>> + '_ {
        RefcountEvent::parser(&stack_matchers)
            .repeated()
            .at_least(1)
            .labelled("refcount event log")
            .map(|events| Self { events })
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::builder()
        .filter_level(LevelFilter::Info)
        .parse_default_env()
        .init();

    let CliArgs { subcommand } = CliArgs::parse();

    match subcommand {
        CliSubcommand::Run { input_dir } => {
            let vs_output_window_text_path = input_dir.join("vs-output-window.txt");
            // TODO: better coercion to `ArcStr` plz
            let vs_output_window_text_path_str = vs_output_window_text_path.to_str().unwrap();
            let vs_output_window_text_path_str =
                ArcStr::from(vs_output_window_text_path_str.to_owned());

            // OPT: buffered reading a chunk at a time for parsing plz
            let vs_output_window_text: ArcStr = {
                fs::read_to_string(&vs_output_window_text_path)
                    .context("failed to read {vs_output_window_text_path:?}")?
                    .into()
            };

            let RefcountEventParsingConfig {
                count_at_start,
                stack_matchers,
            } = {
                let config_path = input_dir.join(config_toml!());
                toml::from_str(&fs::read_to_string(&config_path).with_context(|| {
                    format!(
                        "failed to read configuration from {}",
                        config_path.display()
                    )
                })?)
                .context(concat!(
                    "failed to deserialize `",
                    config_toml!(),
                    "`"
                ))?
            };

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

                    let mut report =
                        Report::build(kind, vs_output_window_text_path_str.clone(), start);
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
            }

            // OPT: We might be paying a minor perf penalty for not pre-allocating here?
            let (events_opt, errs) = RefcountEventLog::parser(&stack_matchers[..])
                .parse_recovery(&vs_output_window_text[..]);

            let reporter = Reporter {
                vs_output_window_text_path_str,
                vs_output_window_text: vs_output_window_text.clone(),
            };

            if !errs.is_empty() {
                // TODO: This is not a great way to determine if something is a warning or error.
                // We should let the error API of the parser report with finer granularity.
                let report_kind = if events_opt.is_some() {
                    ReportKind::Warning
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
                        report.set_message(format!(
                            "Failed to parse refcount event from file: {msg}"
                        ));
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

            let RefcountEventLog { events } =
                events_opt.context("unrecoverable errors encountered while parsing, aborting")?;

            match events.first() {
                Some(Spanned(RefcountEvent::Start { .. }, _)) | None => (),
                Some(Spanned(event, span)) => {
                    reporter.report(ReportKind::Warning, span.start, |report| {
                        let start_event =
                            lazy_format!("{:?}", RefcountEventKind::Start).fg(Color::Green);
                        report.set_message(format!("first refcount event is not a start"));
                        report.add_label(reporter.label(span.clone()).with_message(format!(
                            "expected a {start_event} event, but found this {} event instead",
                            lazy_format!("{:?}", event.kind()).fg(Color::Yellow),
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
            }

            // TODO: handle multiple refcount lifecycles?
            // TODO: use a smaller span for errors, track spans in parsed things
            let mut events = events.iter().skip(1); // skipping the start event we
                                                    // validated earlier
            let mut found_issue = false;
            let mut computed = ComputedRefcount {
                refcount: count_at_start.get(),
                num_unidentified: 0,
                num_dupe_start_events: 0,
            };
            let misbehavior_help_msg = "If you're looking for root causes of program misbehavior, \
                then you may have found a lead your problem!";
            let destructor_event =
                lazy_format!("{:?}", RefcountEventKind::Destructor).fg(Color::Magenta);
            let only_support_one_thing_msg =
                "This tool does not currently support more than a single lifecycle of single \
                refcounted pointer. Make sure you record only a single refcount lifecycle for \
                now.";

            loop {
                // TODO: Unify diagnostic colors for event operation identifiers?
                // TODO: dedupe computed printing
                // OPT: We can probably used direct addition, instead of `checked_add`, since
                // we've already verified that everything can fit in memory.
                match events.next() {
                    // We should already have done something about this, so skippit.
                    Some(Spanned(event, span)) => match event {
                        RefcountEvent::Unidentified { .. } => {
                            computed.num_unidentified =
                                computed.num_unidentified.checked_add(1).unwrap();
                            reporter.report(ReportKind::Error, span.start, |report| {
                                report.set_message(format!("failed to classify event"));
                                report.add_label(reporter.label(span.clone()).with_message(
                                    format!(
                                        "could not classify this event based on its call stack \
                                        from current configuration; refcounts are going to be \
                                        wrong!"
                                    ),
                                ));
                                report.set_help(format!(concat!(
                                    "You should change your `[[stack_matcher]]` configuration \
                                        in your `",
                                    config_toml!(),
                                    "` so that it classifies this call stack correctly."
                                )));
                                report.set_note(format!("computed refcount state: {computed:?}"));
                            });
                            found_issue = true;
                        }
                        RefcountEvent::Start { .. } => {
                            computed.num_dupe_start_events =
                                computed.num_dupe_start_events.checked_add(1).unwrap();
                            reporter.report(ReportKind::Error, span.start, |report| {
                                report.set_message("multiple start events found");
                                report.set_help(
                                    "This tool does not currently support more than a single \
                                    lifecycle of refcounted pointer. Make sure you record only a \
                                    single refcount lifecycle for now.",
                                );
                                report.add_label(
                                    reporter.label(span.clone()).with_message(format!("")),
                                );
                                report.set_note(format!("computed refcount state: {computed:?}"));
                            });
                            found_issue = true;
                        }
                        RefcountEvent::Increment { .. } => {
                            computed.refcount = computed.refcount.checked_add(1).unwrap()
                            // TODO: `INFO`-level logging
                        }
                        RefcountEvent::Decrement { .. } => {
                            computed.refcount = computed.refcount.checked_sub(1).unwrap();
                            // TODO: `INFO`-level logging
                            if computed.refcount == 0 {
                                // TODO: `INFO`-level logging
                                break;
                            }
                        }
                        RefcountEvent::Destructor { .. } => {
                            reporter.report(ReportKind::Warning, span.start, |report| {
                                report.set_message("destructor called before refcount was zero");
                                // TODO: track previous event span?
                                report.add_label(reporter.label(span.clone()).with_message(
                                    format!(
                                        "This {} operation happened while the refcount was still \
                                        positive, which is invalid.",
                                        destructor_event
                                    ),
                                ));
                                report.set_help(misbehavior_help_msg);
                                report.set_note(format!("computed refcount state: {computed:?}"));
                            });
                            found_issue = true;
                            break;
                        }
                    },
                    None => {
                        reporter.report(
                            ReportKind::Warning,
                            vs_output_window_text.len() - 1, // TODO: is this a sensible value?
                            |report| {
                                report.set_message("no destructor found in log");
                                report.set_note(format!("computed refcount state: {computed:?}"));
                                report.set_help(misbehavior_help_msg);
                            },
                        );
                        found_issue = true;
                        break;
                    }
                }
            }

            match events.clone().next() {
                Some(Spanned(RefcountEvent::Destructor { .. }, _)) => {
                    if let Some(Spanned(event, span)) = events.next() {
                        let remaining_event_count = events.count();
                        reporter.report(ReportKind::Error, span.start, |report| {
                            report.set_message(format!(
                                "expected destructor call as final operation"
                            ));
                            report.add_label(reporter.label(span.clone()).with_message(format!(
                                "this {} operation and {remaining_event_count} other refcount \
                                operations were logged after the refcount reached 0 and was \
                                destroyed",
                                lazy_format!("{:?}", event.kind()).fg(Color::Red)
                            )));
                            report.set_note(format!("computed refcount state: {computed:?}"));
                            report.set_help(only_support_one_thing_msg);
                        });
                        found_issue = true;
                    }
                }
                Some(Spanned(event, span)) => {
                    reporter.report(ReportKind::Warning, span.start, |report| {
                        report.set_message(format!("expected destructor call as final operation"));
                        report.add_label(reporter.label(span.clone()).with_message(format!(
                            "this {} operation was called after the refcount was computed to \
                            reach 0, but a {destructor_event} event was expected",
                            lazy_format!("{:?}", event.kind()).fg(Color::Red)
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
                    reporter.report(
                        ReportKind::Warning,
                        vs_output_window_text.len() - 1,
                        |report| {
                            report.set_message(format!("no {destructor_event} found in log"));
                            // format!(
                            //     "expected destructor call as next operation after refcount was \
                            //         computed to reach 0, but no more events are found (computed: \
                            //         {computed:?})"
                            // ),
                            report.set_note(format!("computed refcount state: {computed:?}"));
                            report.set_help(misbehavior_help_msg);
                        },
                    );
                    found_issue = true;
                }
            }

            if found_issue {
                Err(anyhow!(
                    "found one or more issues while analyzing this log; see logs above for more \
                    details"
                ))
            } else {
                Ok(())
            }
        }
    }
}
