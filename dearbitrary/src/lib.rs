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

#[cfg(feature = "derive")]
pub use derive_dearbitrary::*;

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

#[derive(Default)]
pub struct Dearbitrator {
    data: VecDeque<u8>,
}

impl Dearbitrator {
    pub fn new() -> Self {
        Dearbitrator::default()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

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

    pub fn push_rev_iter_first<I: Iterator>(mut iter: I) -> Dearbitrator
    where
        <I as Iterator>::Item: Dearbitrary,
    {
        let mut d = Dearbitrator::new();
        d.push_rev_iter(iter);
        d
        // if let Some(cur) = iter.next() {
        //     let mut d = cur.dearbitrary_first();
        //     for v in iter {
        //         v.dearbitrary(&mut d);
        //     }
        //     d
        // } else {
        //     Dearbitrator::new()
        // }
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

    fn dearbitrary_first(&self) -> Dearbitrator {
        let mut d = Dearbitrator::new();
        self.dearbitrary(&mut d);
        d
    }
}

impl<T: Dearbitrary> Dearbitrary for &T {
    fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
        (*self).dearbitrary(dearbitrator)
    }

    fn dearbitrary_first(&self) -> Dearbitrator {
        (*self).dearbitrary_first()
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

macro_rules! impl_dearbitrary_for_floats {
    ( $( $ty:ident : $unsigned:ty; )* ) => {
        $(
            impl Dearbitrary for $ty {
                fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
                    ((*self).to_bits()).dearbitrary(dearbitrator);
                }
            }

        )*
    }
}

impl_dearbitrary_for_floats! {
    f32: u32;
    f64: u64;
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

            fn dearbitrary_first(&self) -> Dearbitrator {
                let mut dearbitrator = self.$ln.dearbitrary_first();
                $( self.$n.dearbitrary(&mut dearbitrator); )*
                dearbitrator
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

    fn dearbitrary_first(&self) -> Dearbitrator {
        // TODO: check this
        let mut d = if let Some(last) = self.last() {
            last.dearbitrary_first()
        } else {
            Dearbitrator::new()
        };
        self.dearbitrary(&mut d);
        d
    }
}

impl<'a> Dearbitrary for &'a [u8] {
    fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
        dearbitrator.push_bytes(self);
        dearbitrator.push_len(self.len());
    }

    fn dearbitrary_first(&self) -> Dearbitrator {
        let mut d = Dearbitrator::new();
        d.push_bytes(self);
        d
    }
}

impl<A: Dearbitrary> Dearbitrary for Vec<A> {
    fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
        dearbitrator.push_rev_iter(self.iter().rev())
    }

    fn dearbitrary_first(&self) -> Dearbitrator {
        let d = Dearbitrator::push_rev_iter_first(self.iter().rev());
        d
    }
}

impl<K: Dearbitrary, V: Dearbitrary> Dearbitrary for BTreeMap<K, V> {
    fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
        dearbitrator.push_rev_iter(self.iter().rev())
    }

    fn dearbitrary_first(&self) -> Dearbitrator {
        let d = Dearbitrator::push_rev_iter_first(self.iter().rev());
        d
    }
}

impl<A: Dearbitrary> Dearbitrary for BTreeSet<A> {
    fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
        dearbitrator.push_rev_iter(self.iter().rev())
    }

    fn dearbitrary_first(&self) -> Dearbitrator {
        let d = Dearbitrator::push_rev_iter_first(self.iter().rev());
        d
    }
}

impl<A: Dearbitrary> Dearbitrary for BinaryHeap<A> {
    fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
        dearbitrator.push_rev_iter(self.iter().rev())
    }

    fn dearbitrary_first(&self) -> Dearbitrator {
        let d = Dearbitrator::push_rev_iter_first(self.iter().rev());
        d
    }
}

impl<A: Dearbitrary + Eq + ::std::hash::Hash, S: BuildHasher + Default> Dearbitrary
    for HashSet<A, S>
{
    fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
        dearbitrator.push_rev_iter(self.iter()) // order does not matter
    }

    fn dearbitrary_first(&self) -> Dearbitrator {
        let d = Dearbitrator::push_rev_iter_first(self.iter());
        d
    }
}

impl<A: Dearbitrary> Dearbitrary for LinkedList<A> {
    fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
        dearbitrator.push_rev_iter(self.iter()) // order does not matter
    }

    fn dearbitrary_first(&self) -> Dearbitrator {
        let d = Dearbitrator::push_rev_iter_first(self.iter().rev());
        d
    }
}

impl<A: Dearbitrary> Dearbitrary for VecDeque<A> {
    fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
        dearbitrator.push_rev_iter(self.iter()) // order does not matter
    }

    fn dearbitrary_first(&self) -> Dearbitrator {
        let d = Dearbitrator::push_rev_iter_first(self.iter().rev());
        d
    }
}

impl<'a> Dearbitrary for &'a str {
    fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
        self.as_bytes().dearbitrary(dearbitrator)
    }

    fn dearbitrary_first(&self) -> Dearbitrator {
        let d = self.as_bytes().dearbitrary_first();
        d
    }
}

impl Dearbitrary for String {
    fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
        (self as &str).dearbitrary(dearbitrator)
    }

    fn dearbitrary_first(&self) -> Dearbitrator {
        (self as &str).dearbitrary_first()
    }
}

impl Dearbitrary for Box<str> {
    fn dearbitrary(&self, dearbitrator: &mut Dearbitrator) {
        (self as &str).dearbitrary(dearbitrator)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use arbitrary::*;

    macro_rules! assert_dearb_arb_eq {
        ($v:expr, $t:ty) => {{
            // with take rest
            let x: $t = $v;
            let bytes = x.dearbitrary_first().finish();

            let mut u = Unstructured::new(&bytes);
            let y = <$t>::arbitrary_take_rest(u).unwrap();
            assert_eq!(x, y);

            // without take rest
            let x: $t = $v;
            let mut d = Dearbitrator::new();
            x.dearbitrary(&mut d);
            let bytes = d.finish();

            let mut u = Unstructured::new(&bytes);
            let y = <$t>::arbitrary(&mut u).unwrap();
            assert_eq!(x, y);
        }};
    }

    #[test]
    fn dearbitrary_for_integers() {
        assert_dearb_arb_eq!(1 | (2 << 8) | (3 << 16) | (4 << 24), i32);
    }

    #[test]
    fn dearbitrary_for_bytes() {
        assert_dearb_arb_eq!(&[1, 2, 3, 4], &[u8]);
    }

    #[test]
    fn dearbitrary_collection() {
        assert_dearb_arb_eq!(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3], &[u8]);
        assert_dearb_arb_eq!(vec![2, 4, 6, 8, 1], Vec<u16>);
        assert_dearb_arb_eq!(vec![84148994], Vec<u32>);
        assert_dearb_arb_eq!(vec![123; 100], Vec<usize>);
        assert_dearb_arb_eq!(
            "\x01\x02\x03\x04\x05\x06\x07\x08\x09\x01\x02\x03".to_string(),
            String
        );
    }

    #[test]
    fn test_multiple_vecs_i32() {
        assert_dearb_arb_eq!((vec![1, 2, 3], vec![10, 20, 21]), (Vec<i32>, Vec<i32>));
    }

    #[test]
    fn test_multiple_vecs_u8() {
        assert_dearb_arb_eq!((vec![1, 2, 3, 4], vec![10, 20, 21]), (Vec<u8>, Vec<u8>));
    }

    #[test]
    fn test_optional() {
        assert_dearb_arb_eq!(
            Some((vec![1, 0xfffffff_u64, 3], 0xfffffff_u64, vec![10, 20])),
            Option<(Vec<u64>, u64, Vec<u8>)>
        );
    }

    #[test]
    fn test_integers() {
        assert_dearb_arb_eq!(-0x12345678_i32, i32);
        assert_dearb_arb_eq!(0x12345678_u64, u64);
        assert_dearb_arb_eq!(1.123f64, f64);
        assert_dearb_arb_eq!(1.123f32, f32);
        assert_dearb_arb_eq!(255, u8);
        assert_dearb_arb_eq!(0x12345678, isize);
        assert_dearb_arb_eq!(0x12345678, usize);
    }

    #[test]
    fn test_strings() {
        assert_dearb_arb_eq!("ABCDEFG", &str);
        assert_dearb_arb_eq!("ABCDEFG".to_string(), String);
        assert_dearb_arb_eq!("ABCDEFG".into(), Box<str>);
    }

    #[test]
    fn test_deep_nest() {
        assert_dearb_arb_eq!(
            vec![(10u8, vec![10f32, 20f32]), (12u8, vec![100f32, 10000f32])],
            Vec<(u8, Vec<f32>)>
        );
        assert_dearb_arb_eq!(
            vec![(10u64, vec![10f32, 20f32]), (12u64, vec![100f32, 10000f32])],
            Vec<(u64, Vec<f32>)>
        );
    }
}
