use std::ops::Range;

#[derive(Clone, Debug, PartialEq)]
pub struct Spanned<T>(pub T, pub Range<usize>);

impl<T> Spanned<T> {
    pub fn new(t: T, span: Range<usize>) -> Self {
        Self(t, span)
    }

    pub fn into_parts(self) -> (T, Range<usize>) {
        let Self(inner, span) = self;
        (inner, span)
    }

    pub fn as_parts(&self) -> (&T, &Range<usize>) {
        let Self(inner, span) = self;
        (inner, span)
    }

    pub fn map<U, F>(self, f: F) -> Spanned<U>
    where
        F: FnOnce(T) -> U,
    {
        let Self(inner, span) = self;
        Spanned(f(inner), span)
    }

    pub fn as_inner(&self) -> &T {
        let Self(inner, _span) = self;
        inner
    }

    pub fn as_ref(&self) -> Spanned<&T> {
        let Self(inner, span) = self;
        Spanned(inner, span.clone())
    }
}
