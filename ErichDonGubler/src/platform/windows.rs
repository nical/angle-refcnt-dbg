use std::fmt::Display;

use chumsky::{
    error::SimpleReason,
    prelude::Simple,
    primitive::{choice, end, filter, just},
    text::newline,
    Parser,
};
use format::lazy_format;
use knuffel::{ast::SpannedNode, decode::Context, errors::DecodeError, traits::ErrorSpan, Decode};
use miette::miette;

use crate::{
    matcher,
    state_machine::{Address, CallStack, StackFrame, SymbolicatedStackFrame},
};

use super::{Platform, PlatformKdl, PlatformLogSyntax};

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct Windows;

impl Windows {
    fn address_parser() -> impl Parser<char, Address, Error = Simple<char>> {
        u64_from_all_digits().map(Address::new)
    }

    fn display_address(address: &Address) -> impl Display + '_ {
        lazy_format!("{address:?}")
    }

    fn stack_frame_parser() -> impl Parser<char, StackFrame, Error = Simple<char>> {
        let external_code = Windows::address_parser()
            .map(|address| StackFrame::ExternalCode { address })
            .labelled("external code");

        let symbolicated = Self::symbolicated_stack_frame_parser().map(StackFrame::Symbolicated);

        choice((external_code, symbolicated))
            .labelled("inner stack frame line")
            .delimited_by(just("\t"), newline())
            .labelled("full stack frame line")
    }

    fn symbolicated_stack_frame_parser(
    ) -> impl Parser<char, SymbolicatedStackFrame, Error = Simple<char>> {
        fn non_newline(c: &char) -> bool {
            let mut buf = [0; 4];
            let s = &*c.encode_utf8(&mut buf);
            end::<Simple<char>>().parse(s).is_err() && newline::<Simple<char>>().parse(s).is_err()
        }
        let chars_until = |pred: fn(&char) -> bool| {
            filter(pred)
                .repeated()
                .at_least(1)
                // TODO: max bounds?
                // OPT: eww, y u no `str`?
                .map(String::from_iter)
        };
        // See also
        // <https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/symbol-syntax-and-symbol-matching>.
        let module = chars_until(|c| non_newline(c) && *c != '!')
            .then_ignore(just("!"))
            .labelled("module");
        let symbol_name = chars_until(non_newline).labelled("symbol name");
        module
            .then(symbol_name)
            .map(|(module, symbol_name)| SymbolicatedStackFrame {
                module,
                symbol_name,
            })
            .labelled("symbolicated stack frame")
    }

    fn call_stack_parser() -> impl Parser<char, CallStack, Error = Simple<char>> {
        // OPT: We're probably paying a non-trivial amount here for parsing dozens of frames
        // without trying to set capacity first.
        Self::stack_frame_parser()
            .repeated()
            .at_least(1)
            .then_ignore(just("\t").then(newline()))
            .labelled("`$CALLSTACK` output")
            .map(|frames| CallStack { frames })
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

    insta::assert_debug_snapshot!(Windows::call_stack_parser().parse(data).unwrap());
}

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

impl PlatformLogSyntax for Windows {
    fn address_parser(&self) -> Box<dyn Parser<char, Address, Error = Simple<char>>> {
        todo!()
    }

    fn display_address(&self) -> Box<dyn Display + '_> {
        todo!()
    }

    fn stack_frame_parser(&self) -> Box<dyn Parser<char, StackFrame, Error = Simple<char>>> {
        Box::new(Self::stack_frame_parser())
    }

    fn display_stack_frame(&self, frame: &StackFrame) -> Box<dyn Display + '_> {
        todo!()
    }

    fn refcount_address_count_pair_parser(
        &self,
    ) -> Box<dyn Parser<char, (Address, u64), Error = Simple<char>>> {
        Box::new(u64_address_value_debug_pair())
    }

    fn call_stack_parser(&self) -> Box<dyn Parser<char, CallStack, Error = Simple<char>>> {
        Box::new(Self::call_stack_parser())
    }
}

fn u64_address_value_debug_pair() -> impl Parser<char, (Address, u64), Error = Simple<char>> {
    just("0x")
        .ignore_then(Windows::address_parser().labelled("changed refcount address"))
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

#[test]
fn parse_symbolicated_stack_frames() {
    assert_eq!(
        Windows::stack_frame_parser().parse("\ta!b\n").unwrap(),
        StackFrame::Symbolicated(SymbolicatedStackFrame {
            module: "a".to_string(),
            symbol_name: "b".to_string(),
        })
    );
    assert_eq!(
        Windows::stack_frame_parser().parse(
            "\td3d11_3SDKLayers.dll!CLayeredObject<NDebug::CDevice>::CContainedObject::Release\n"
        )
        .unwrap(),
        StackFrame::Symbolicated(SymbolicatedStackFrame {
            module: "d3d11_3SDKLayers.dll".to_string(),
            symbol_name: "CLayeredObject<NDebug::CDevice>::CContainedObject::Release"
                .to_string(),
        })
    );
    assert!(Windows::stack_frame_parser().parse("\td3d11_").is_err());
    assert!(Windows::stack_frame_parser().parse("\td3d11_\n").is_err());
    assert!(Windows::stack_frame_parser().parse("\td3d11!\n").is_err());
    assert!(Windows::stack_frame_parser().parse("\t!\n").is_err());
    assert!(Windows::stack_frame_parser().parse("\t!asdf\n").is_err());
}
