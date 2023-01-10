use std::{
    cmp::Ordering,
    fmt::{self, Debug, Formatter},
    fs,
    ops::Range,
    path::PathBuf,
};

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

use crate::config::{RefcountEventParsingConfig, RefcountOpClassifier};

mod config;

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

macro_rules! start_of_event {
    () => {
        "--!! "
    };
}
const START_OF_EVENT: &str = start_of_event!();

macro_rules! config_basename {
    () => {
        "antileak.kdl"
    };
}

#[derive(Debug, PartialEq)]
pub struct Spanned<T>(T, Range<usize>);

#[derive(Debug)]
struct RefcountEvent {
    address: Address,
    /// TODO: This doesn't take into account the size of the refcount maybe _not_ being 64 bits. We
    /// should do that, if this ever needs to be robust.
    count: u64,
    callstack: CallStack,
    kind: RefcountEventKind,
}

#[derive(Debug)]
enum RefcountEventKind {
    Start { variable_name: String },
    Modify { kind: RefcountModifyKind },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RefcountModifyKind {
    Increment,
    Decrement,
    Destructor,
    Unidentified,
}

#[derive(Clone, Debug)]
struct ComputedRefcount {
    refcount: u64,
    num_unidentified: u64,
    num_dupe_start_events: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RefcountEventKindName {
    Start,
    Increment,
    Decrement,
    Destructor,
    Unidentified,
}

impl From<RefcountModifyKind> for RefcountEventKindName {
    fn from(value: RefcountModifyKind) -> Self {
        match value {
            RefcountModifyKind::Increment => Self::Increment,
            RefcountModifyKind::Decrement => Self::Decrement,
            RefcountModifyKind::Destructor => Self::Destructor,
            RefcountModifyKind::Unidentified => Self::Unidentified,
        }
    }
}

impl RefcountEvent {
    pub fn kind_name(&self) -> RefcountEventKindName {
        match &self.kind {
            RefcountEventKind::Start { .. } => RefcountEventKindName::Start,
            &RefcountEventKind::Modify { kind, .. } => kind.into(),
        }
    }

    fn parser(
        stack_matchers: &[RefcountOpClassifier],
    ) -> impl Parser<char, Spanned<Self>, Error = Simple<char>> + '_ {
        take_until(just(START_OF_EVENT).labelled(concat!(
            "next refcount event sentinel (`",
            start_of_event!(),
            "`)"
        )))
        .ignore_then(
            choice((
                just("Ref count was changed for ")
                    .ignore_then(u64_address_value_debug_pair())
                    .then_ignore(just(":"))
                    .then(CallStack::parser().labelled("call stack"))
                    .map(|((address, count), callstack)| {
                        let kind = stack_matchers
                            .iter()
                            .find_map(|matcher| matcher.matches(&callstack))
                            .map(Into::into)
                            .unwrap_or(RefcountModifyKind::Unidentified);
                        Self {
                            address,
                            count,
                            callstack,
                            kind: RefcountEventKind::Modify { kind },
                        }
                    }),
                just("Starting to track refs for `")
                    .ignore_then(ident().labelled("COM object identifier"))
                    .then_ignore(just("` at "))
                    .then(u64_address_value_debug_pair())
                    .then_ignore(just(":"))
                    .then(CallStack::parser().labelled("call stack"))
                    .map(|((variable_name, (address, count)), callstack)| Self {
                        kind: RefcountEventKind::Start { variable_name },
                        address,
                        count,
                        callstack,
                    }),
            ))
            .labelled("event body")
            .map_with_span(Spanned),
        )
        .labelled("refcount event")
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

#[derive(Debug, Eq, PartialEq)]
enum StackFrame {
    ExternalCode { address: Address },
    Symbolicated { module: String, symbol_name: String },
}

fn u64_address_value_debug_pair() -> impl Parser<char, (Address, u64), Error = Simple<char>> {
    just("0x")
        .ignore_then(Address::parser().labelled("changed refcount address"))
        .then_ignore(just(" {0x"))
        .then(u64_from_all_digits())
        .then_ignore(just("}"))
}

fn u64_from_all_digits() -> impl Parser<char, u64, Error = Simple<char>> {
    filter(char::is_ascii_hexdigit)
        .repeated()
        .exactly(16)
        .labelled("full ASCII hex digits of `u64`")
        // OPT: s/String/&str?
        .map(|digits| String::from_iter(digits.into_iter()))
        .map(|digits| u64::from_str_radix(&digits, 16).unwrap())
}

#[derive(Eq, PartialEq)]
struct Address {
    value: u64,
}

impl Debug for Address {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Self { value } = self;
        write!(f, "{value:#010X}")
    }
}

impl Address {
    fn parser() -> impl Parser<char, Self, Error = Simple<char>> {
        u64_from_all_digits().map(|value| Self { value })
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
                .map(String::from_iter)
        };
        let module = chars_until(|c| non_newline(c) && *c != '!')
            .then_ignore(just("!"))
            .labelled("module");
        let symbol_name = chars_until(non_newline).labelled("symbol name");
        let symbolicated_stack_frame =
            module
                .then(symbol_name)
                .map(|(module, symbol_name)| Self::Symbolicated {
                    module,
                    symbol_name,
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
        StackFrame::Symbolicated {
            module: "a".to_string(),
            symbol_name: "b".to_string(),
        }
    );
    assert_eq!(
        StackFrame::parser().parse(
            "\td3d11_3SDKLayers.dll!CLayeredObject<NDebug::CDevice>::CContainedObject::Release\n"
        )
        .unwrap(),
        StackFrame::Symbolicated {
            module: "d3d11_3SDKLayers.dll".to_string(),
            symbol_name: "CLayeredObject<NDebug::CDevice>::CContainedObject::Release"
                .to_string(),
        }
    );
    assert!(StackFrame::parser().parse("\td3d11_").is_err());
    assert!(StackFrame::parser().parse("\td3d11_\n").is_err());
    assert!(StackFrame::parser().parse("\td3d11!\n").is_err());
    assert!(StackFrame::parser().parse("\t!\n").is_err());
    assert!(StackFrame::parser().parse("\t!asdf\n").is_err());
}

// OPT: intern frame names, source names?
#[derive(Debug, Eq, PartialEq)]
struct SourceLocation {
    file_path: PathBuf,
    line: u64,
}

struct RefcountEventLog {
    events: Vec<Spanned<RefcountEvent>>,
}

impl RefcountEventLog {
    fn parser(
        stack_matchers: &[RefcountOpClassifier],
    ) -> impl Parser<char, Self, Error = Simple<char>> + '_ {
        RefcountEvent::parser(stack_matchers)
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
            .context("failed to read {vs_output_window_text_path:?}")?
            .into()
    };

    let RefcountEventParsingConfig { op_classifiers } = {
        let config_path = input_dir.join(config_basename!());
        knuffel::parse(
            &config_path.to_str().unwrap(),
            &fs::read_to_string(&config_path).with_context(|| {
                format!(
                    "failed to read configuration from {}",
                    config_path.display()
                )
            })?,
        )
        .map_err(|e| {
            eprintln!("{}", miette::Report::new(e));

            anyhow!(concat!(
                "failed to deserialize `",
                config_basename!(),
                "`(see above for more details)"
            ))
        })?
    };

    // OPT: We might be paying a minor perf penalty for not pre-allocating here?
    let (events_opt, errs) =
        RefcountEventLog::parser(&op_classifiers[..]).parse_recovery(&vs_output_window_text[..]);

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
    }

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

    match subcommand {
        CliSubcommand::Lint => {
            let start_event = lazy_format!("{:?}", RefcountEventKindName::Start).fg(Color::Green);

            let RefcountEventLog { events } =
                events_opt.context("unrecoverable errors encountered while parsing, aborting")?;

            let mut found_issue = false;
            let count_at_start = match events.first() {
                Some(Spanned(
                    RefcountEvent {
                        kind: RefcountEventKind::Start { .. },
                        count,
                        ..
                    },
                    span,
                )) => {
                    if *count == 0 {
                        reporter.report(ReportKind::Error, span.start, |report| {
                            report.set_message("refcount start event has count of 0");
                            report.add_label(reporter.label(span.clone()).with_message(format!(
                                "this {start_event} event has a refcount of 0, which is almost \
                                certainly wrong",
                            )));
                        });
                        found_issue = true;
                    }
                    *count
                }
                None => {
                    log::info!("no events found, so log is trivially linted");
                    return Ok(());
                }
                Some(Spanned(event, span)) => {
                    reporter.report(ReportKind::Warning, span.start, |report| {
                        let start_event =
                            lazy_format!("{:?}", RefcountEventKindName::Start).fg(Color::Green);
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

            // TODO: handle multiple refcount lifecycles?
            // TODO: use a smaller span for errors, track spans in parsed things
            let mut events = events.iter().skip(1); // skipping the start event we
                                                    // validated earlier
            let mut computed = ComputedRefcount {
                refcount: count_at_start,
                num_unidentified: 0,
                num_dupe_start_events: 0,
            };
            let misbehavior_help_msg = "If you're looking for root causes of program misbehavior, \
                then you may have found a lead!";
            let destructor_event =
                lazy_format!("{:?}", RefcountEventKindName::Destructor).fg(Color::Magenta);
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
                    Some(Spanned(event, span)) => match event {
                        RefcountEvent {
                            kind: RefcountEventKind::Start { .. },
                            ..
                        } => {
                            computed.num_dupe_start_events =
                                computed.num_dupe_start_events.checked_add(1).unwrap();
                            reporter.report(ReportKind::Error, span.start, |report| {
                                report.set_message("multiple start events found");
                                report.set_help(only_support_one_thing_msg);
                                report.add_label(reporter.label(span.clone()).with_message(
                                    "this {start_event} event occurs after the first one (TODO: \
                                    span)",
                                ));
                                report.set_note(computed_note(&computed));
                            });
                            found_issue = true;
                        }
                        RefcountEvent {
                            count,
                            kind: RefcountEventKind::Modify { kind, .. },
                            ..
                        } => {
                            match kind {
                                RefcountModifyKind::Unidentified { .. } => {
                                    computed.num_unidentified =
                                        computed.num_unidentified.checked_add(1).unwrap();
                                    let suggested_match = match *count {
                                        c if c == computed.refcount - 1 => Some("decrement"),
                                        c if c == computed.refcount.checked_add(1).unwrap() => {
                                            Some("increment")
                                        }
                                        _ => None,
                                    };
                                    reporter.report(ReportKind::Error, span.start, |report| {
                                        report.set_message("failed to classify event");
                                        report.add_label(
                                            reporter.label(span.clone()).with_message(
                                                "could not classify this event based on its call
                                                stack from current configuration; refcounts are \
                                                going to be wrong!",
                                            ),
                                        );
                                        report.set_help(format!(
                                            concat!(
                                                "You should change your stack matching \
                                                configuration in your `",
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
                                RefcountModifyKind::Increment { .. } => {
                                    computed.refcount = computed.refcount.checked_add(1).unwrap()
                                    // TODO: `INFO`-level logging
                                }
                                RefcountModifyKind::Decrement { .. } => {
                                    computed.refcount = computed.refcount.checked_sub(1).unwrap();
                                    // TODO: `INFO`-level logging
                                    if computed.refcount == 0 {
                                        // TODO: `INFO`-level logging
                                        break;
                                    }
                                }
                                RefcountModifyKind::Destructor { .. } => {
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
                                report.set_note(computed_note(&computed));
                                });
                                    found_issue = true;
                                    break;
                                }
                            };
                            if computed.refcount != *count {
                                reporter.report(ReportKind::Error, span.start, |report| {
                                    report.set_message(
                                        "refcount in log diverges from computed value",
                                    );
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
                                    the logged code."
                                );
                                    report.set_note(computed_note(&computed));
                                });
                                found_issue = true;
                            }
                        }
                    },
                    None => {
                        reporter.report(
                            ReportKind::Warning,
                            vs_output_window_text.len() - 1, // TODO: is this a sensible value?
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
                    RefcountEvent {
                        kind:
                            RefcountEventKind::Modify {
                                kind: RefcountModifyKind::Destructor,
                            },
                        ..
                    },
                    _, // TODO: use this span as part of a label
                )) => {
                    if let Some(Spanned(event, span)) = events.nth(1) {
                        let extra_maybe = if event.kind_name() == RefcountEventKindName::Destructor
                        {
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
                    reporter.report(
                        ReportKind::Warning,
                        vs_output_window_text.len() - 1,
                        |report| {
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
        CliSubcommand::ShowExec => {
            let mut event_iter = events_opt
                .iter()
                .flat_map(|log| log.events.iter().map(|Spanned(event, _span)| event));
            let mut previous_event = match event_iter.next() {
                None => {
                    println!("! no events found");
                    return Ok(());
                }
                Some(event) => event,
            };

            let wind_up = |idx, event: &RefcountEvent| {
                let frames = &event.callstack.frames;
                for (idx, frame) in frames.iter().rev().enumerate().skip(idx) {
                    let stack_frame = lazy_format!(|f| match frame {
                        StackFrame::ExternalCode { address } => write!(f, "{address:?}"),
                        StackFrame::Symbolicated {
                            module,
                            symbol_name,
                        } => write!(f, "{module}!{symbol_name}"),
                    });
                    println!("{:>frames_up$}\\ {stack_frame}", "", frames_up = idx);
                }
                println!("{:>frames_up$}|", "", frames_up = frames.len());
                println!(
                    "{:>frames_up$}* {:?} of {:?} -> {}",
                    "",
                    event.kind_name(),
                    event.address,
                    event.count,
                    frames_up = frames.len()
                );
            };
            let wind_down = |frames: &[StackFrame], idx| match frames.len().cmp(&idx) {
                Ordering::Less => unreachable!(),
                Ordering::Equal => (),
                Ordering::Greater => {
                    println!(
                        "{0:>space$}{0:_>frames_down$}|",
                        "",
                        space = idx,
                        frames_down = frames.len().checked_sub(idx).unwrap()
                    );
                }
            };

            wind_up(0, previous_event);
            for next_event in event_iter {
                let shared_idx = |prev: &[_], next: &[_]| {
                    prev.iter()
                        .rev()
                        .zip(next.iter().rev())
                        .position(|(f1, f2)| f1 != f2)
                        .unwrap_or(prev.len())
                };
                let previous_frames = &previous_event.callstack.frames;
                let next_frames = &next_event.callstack.frames;
                let common_frames_idx = shared_idx(previous_frames, &next_frames);
                wind_down(previous_frames, common_frames_idx);
                wind_up(common_frames_idx, next_event);
                previous_event = next_event;
            }
            wind_down(&previous_event.callstack.frames, 0);

            Ok(())
        }
    }
}
