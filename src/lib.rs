// Copyright © 2019 The Rust Fuzz Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The `Dearbitrary` trait crate.
//!
//! This trait provides an [`Dearbitrary`](./trait.Dearbitrary.html) trait.

#![deny(bad_style)]
// #![deny(missing_docs)]
#![deny(future_incompatible)]
#![deny(nonstandard_style)]
#![deny(rust_2018_compatibility)]
#![deny(rust_2018_idioms)]
// #![deny(unused)]
#![allow(unused)]

#[cfg(feature = "derive_arbitrary")]
pub use derive_arbitrary::*;

mod error;
pub use error::*;

use core::array;
use core::cell::{Cell, RefCell, UnsafeCell};
use core::iter;
use core::mem;
use core::num::{NonZeroI128, NonZeroI16, NonZeroI32, NonZeroI64, NonZeroI8, NonZeroIsize};
use core::num::{NonZeroU128, NonZeroU16, NonZeroU32, NonZeroU64, NonZeroU8, NonZeroUsize};
use core::ops::{Range, RangeBounds, RangeFrom, RangeInclusive, RangeTo, RangeToInclusive};
use core::str;
use core::time::Duration;
use std::borrow::{Cow, ToOwned};
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet, LinkedList, VecDeque};
use std::ffi::{CString, OsString};
use std::hash::BuildHasher;
use std::net::{Ipv4Addr, Ipv6Addr};
use std::ops::Bound;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicUsize};
use std::sync::{Arc, Mutex};

pub struct Dearbitrator {
    pub data: VecDeque<u8>,
}

impl Dearbitrator {
    pub fn new() -> Self {
        Dearbitrator {
            data: VecDeque::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    // pub fn dearbitrary_len(&mut self, len: usize) {
    //
    // }

    pub fn push_rev_iter<I: Iterator>(&mut self, iter: I)
    where
        <I as Iterator>::Item: Dearbitrary,
    {
        false.dearbitrary(self);
        for v in iter {
            v.dearbitrary(self);
            true.dearbitrary(self);
        }
    }

    pub fn push_rev_iter_last<I: Iterator>(&mut self, iter: I)
    where
        <I as Iterator>::Item: Dearbitrary,
    {
        for v in iter {
            v.dearbitrary(self);
        }
    }

    pub fn push_bytes(&mut self, data: &[u8]) {
        for b in data.iter().rev() {
            self.data.push_front(*b)
        }
    }

    pub fn push_len(&mut self, len: usize) {

        if self.data.len() as u64 <= std::u8::MAX as u64 {
            let len = len as u8;
            for b in len.to_be_bytes() {
                self.data.push_back(b);
            }
        } else if self.data.len() as u64 <= std::u16::MAX as u64 {
            let len = len as u16;
            for b in len.to_be_bytes() {
                self.data.push_back(b);
            }
        } else if self.data.len() as u64 <= std::u32::MAX as u64 {
            let len = len as u32;
            for b in len.to_be_bytes() {
                self.data.push_back(b);
            }
        } else {
            let len = len as u64;
            for b in len.to_be_bytes() {
                self.data.push_back(b);
            }
        };

    }

    pub fn finish(self) -> Vec<u8> {
        self.data.into()
    }
}

pub trait Dearbitrary {
    fn dearbitrary(&self, dearbitrator: &mut Dearbitrator);

    fn dearbitraty_full(&self) -> Vec<u8> {
        let mut d = Dearbitrator::new();
        self.dearbitrary(&mut d);
        d.finish()
    }
}


impl<T: Dearbitrary> Dearbitrary for &T {
    fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
        (*self).dearbitrary(dearbitrator)
    }
}



impl Dearbitrary for () {
    fn dearbitrary(&self, _dearbitrator: &mut Dearbitrator) {}
}

impl Dearbitrary for bool {
    fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
        (*self as u8).dearbitrary(dearbitrator);
    }
}


macro_rules! impl_dearbitrary_for_integers {
    ( $( $ty:ty: $unsigned:ty; )* ) => {
        $(
            impl Dearbitrary for $ty {
                fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
                    let mut buf = [0; mem::size_of::<$ty>()];
                    let x: $unsigned = *self as $unsigned;
                    for i in 0..mem::size_of::<$ty>() {
                        buf[i] = ((x >> ( i * 8 )) & 0xff) as u8;
                    }
                    dearbitrator.push_bytes(&buf);
                }
            }
        )*
    }
}

impl_dearbitrary_for_integers! {
    u8: u8;
    u16: u16;
    u32: u32;
    u64: u64;
    u128: u128;
    usize: usize;
    i8: u8;
    i16: u16;
    i32: u32;
    i64: u64;
    i128: u128;
    isize: usize;
}


impl Dearbitrary for char {
    fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
        (*self as u32).dearbitrary(dearbitrator);
    }
}

impl Dearbitrary for AtomicBool {
    fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
        (self.load(std::sync::atomic::Ordering::SeqCst)).dearbitrary(dearbitrator);
    }
}

impl Dearbitrary for AtomicIsize {
    fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
        (self.load(std::sync::atomic::Ordering::SeqCst)).dearbitrary(dearbitrator);
    }
}

impl Dearbitrary for AtomicUsize {
    fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
        (self.load(std::sync::atomic::Ordering::SeqCst)).dearbitrary(dearbitrator);
    }
}

pub(crate) fn bounded_range<CB, I, R>(bounds: (I, I), cb: CB) -> R
where
    CB: Fn((I, I)) -> R,
    I: PartialOrd,
    R: RangeBounds<I>,
{
    let (mut start, mut end) = bounds;
    if start > end {
        mem::swap(&mut start, &mut end);
    }
    cb((start, end))
}

pub(crate) fn unbounded_range<CB, I, R>(bound: I, cb: CB) -> R
where
    CB: Fn(I) -> R,
    R: RangeBounds<I>,
{
    cb(bound)
}

impl<A: Dearbitrary> Dearbitrary for Option<A> {
    fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
        match self {
            Some(v) => {
                v.dearbitrary(dearbitrator);
                true.dearbitrary(dearbitrator);
            }
            None => false.dearbitrary(dearbitrator),
        }
    }
}

impl<A: Dearbitrary, B: Dearbitrary> Dearbitrary for std::result::Result<A, B> {
    fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
        match self {
            Ok(v) => {
                v.dearbitrary(dearbitrator);
                true.dearbitrary(dearbitrator);
            }
            Err(v) => {
                v.dearbitrary(dearbitrator);
                false.dearbitrary(dearbitrator);
            }
        }
    }
}

macro_rules! arbitrary_tuple {
    () => {};
    ($ln: tt $last: ident $($n: tt $xs: ident)*) => {
        arbitrary_tuple!($($n $xs)*);

        impl<$($xs: Dearbitrary,)* $last: Dearbitrary> Dearbitrary for ($($xs,)* $last,) {
            fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
                self.$ln.dearbitrary(dearbitrator);
                $( self.$n.dearbitrary(dearbitrator); )*
            }
        }


    };
}

arbitrary_tuple!(25 A 24 B 23 C 22 D 21 E 20 F 19 G 18 H 17 I 16 J 15 K 14 L 13 M 12 N 11 O 10 P 9 Q 8 R 7 S 6 T 5 U 4 V 3 W 2 X 1 Y 0 Z);

// Helper to safely create arrays since the standard library doesn't
// provide one yet. Shouldn't be necessary in the future.
struct ArrayGuard<T, const N: usize> {
    dst: *mut T,
    initialized: usize,
}

impl<T, const N: usize> Drop for ArrayGuard<T, N> {
    fn drop(&mut self) {
        debug_assert!(self.initialized <= N);
        let initialized_part = core::ptr::slice_from_raw_parts_mut(self.dst, self.initialized);
        unsafe {
            core::ptr::drop_in_place(initialized_part);
        }
    }
}

fn try_create_array<F, T, const N: usize>(mut cb: F) -> Result<[T; N]>
where
    F: FnMut(usize) -> Result<T>,
{
    let mut array: mem::MaybeUninit<[T; N]> = mem::MaybeUninit::uninit();
    let array_ptr = array.as_mut_ptr();
    let dst = array_ptr as _;
    let mut guard: ArrayGuard<T, N> = ArrayGuard {
        dst,
        initialized: 0,
    };
    unsafe {
        for (idx, value_ptr) in (*array.as_mut_ptr()).iter_mut().enumerate() {
            core::ptr::write(value_ptr, cb(idx)?);
            guard.initialized += 1;
        }
        mem::forget(guard);
        Ok(array.assume_init())
    }
}

impl<T: Dearbitrary, const N: usize> Dearbitrary for [T; N] {
    fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
        for v in self.iter().rev() {
            v.dearbitrary(dearbitrator)
        }
    }
}

impl<'a> Dearbitrary for &'a [u8] {
    fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
        dearbitrator.push_bytes(self);
        dearbitrator.push_len(self.len());
    }
}

impl<A: Dearbitrary> Dearbitrary for Vec<A> {
    fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
        dearbitrator.push_rev_iter(self.iter().rev())
    }
}

impl<K: Dearbitrary, V: Dearbitrary> Dearbitrary for BTreeMap<K, V> {
    fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
        dearbitrator.push_rev_iter(self.iter().rev())
    }
}

impl<A: Dearbitrary> Dearbitrary for BTreeSet<A> {
    fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
        dearbitrator.push_rev_iter(self.iter().rev())
    }
}

impl<A: Dearbitrary> Dearbitrary for BinaryHeap<A> {
    fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
        dearbitrator.push_rev_iter(self.iter().rev())
    }
}


impl<A: Dearbitrary + Eq + ::std::hash::Hash, S: BuildHasher + Default> Dearbitrary for HashSet<A, S> {
    fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
        dearbitrator.push_rev_iter(self.iter()) // order does not matter
    }
}

impl<A: Dearbitrary> Dearbitrary for LinkedList<A> {
    fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
        dearbitrator.push_rev_iter(self.iter()) // order does not matter
    }
}

impl<A: Dearbitrary> Dearbitrary for VecDeque<A> {
    fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
        dearbitrator.push_rev_iter(self.iter()) // order does not matter
    }
}

impl<'a> Dearbitrary for &'a str {
    fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
        self.as_bytes().dearbitrary(dearbitrator)
    }
}

impl Dearbitrary for String {
    fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
        (&self as &str).dearbitrary(dearbitrator)
    }
}

impl Dearbitrary for Box<str> {
    fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
        (&self as &str).dearbitrary(dearbitrator)
    }
}

/// Multiple conflicting arbitrary attributes are used on the same field:
/// ```compile_fail
/// #[derive(::arbitrary::Arbitrary)]
/// struct Point {
///     #[arbitrary(value = 2)]
///     #[arbitrary(value = 2)]
///     x: i32,
/// }
/// ```
///
/// An unknown attribute:
/// ```compile_fail
/// #[derive(::arbitrary::Arbitrary)]
/// struct Point {
///     #[arbitrary(unknown_attr)]
///     x: i32,
/// }
/// ```
///
/// An unknown attribute with a value:
/// ```compile_fail
/// #[derive(::arbitrary::Arbitrary)]
/// struct Point {
///     #[arbitrary(unknown_attr = 13)]
///     x: i32,
/// }
/// ```
///
/// `value` without RHS:
/// ```compile_fail
/// #[derive(::arbitrary::Arbitrary)]
/// struct Point {
///     #[arbitrary(value)]
///     x: i32,
/// }
/// ```
///
/// `with` without RHS:
/// ```compile_fail
/// #[derive(::arbitrary::Arbitrary)]
/// struct Point {
///     #[arbitrary(with)]
///     x: i32,
/// }
/// ```
#[cfg(all(doctest, feature = "derive"))]
pub struct CompileFailTests;
