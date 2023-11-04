#![cfg(feature = "derive")]
// Various structs/fields that we are deriving `Arbitrary` for aren't actually
// used except to exercise the derive.
#![allow(dead_code)]

use arbitrary::*;
use dearbitrary::*;

macro_rules! assert_dearb_arb_eq {
    ($v:expr, $t:ty) => {{
        // with take rest
        let x: $t = $v;
        let bytes = x.dearbitrary_first().finish();

        let u = Unstructured::new(&bytes);
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

#[derive(Copy, Clone, Debug, Eq, PartialEq, Arbitrary, Dearbitrary)]
pub struct Rgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

#[test]
fn struct_with_named_fields() {
    assert_dearb_arb_eq!(
        Rgb {
            r: 10,
            b: 20,
            g: 30,
        },
        Rgb
    );
}

#[derive(Clone, Debug, Eq, PartialEq, Arbitrary, Dearbitrary)]
pub struct ArrStruct {
    pub o: Vec<Vec<u32>>,
    pub r: Vec<u64>,
    pub g: [Rgb; 2],
    pub b: Vec<u8>,
}

#[test]
fn struct_with_multiple_arrays() {
    assert_dearb_arb_eq!(
        ArrStruct {
            o: vec![vec![], vec![10; 2000], vec![], vec![100]],
            r: vec![10, 20, 30, 40],
            g: [Rgb { r: 0, g: 1, b: 5 }, Rgb { r: 0, g: 1, b: 5 }],
            b: vec![30, 40],
        },
        ArrStruct
    );
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Arbitrary, Dearbitrary)]
struct MyTupleStruct(u8, bool);

#[test]
fn tuple_struct() {
    assert_dearb_arb_eq!(MyTupleStruct(43, true), MyTupleStruct);
    assert_dearb_arb_eq!(MyTupleStruct(10, false), MyTupleStruct);
}

#[derive(Clone, Debug, Eq, PartialEq, Arbitrary, Dearbitrary)]
struct EndingInVec(u8, bool, u32, Vec<u16>);
#[derive(Clone, Debug, Eq, PartialEq, Arbitrary, Dearbitrary)]
struct EndingInString(u8, bool, u32, String);

#[test]
fn test_take_rest() {
    assert_dearb_arb_eq!(
        EndingInVec(1, false, 0x4030201, vec![0x605, 0x807]),
        EndingInVec
    );
    assert_dearb_arb_eq!(
        EndingInString(1, false, 0x4030201, "\x05\x06\x07\x08".to_string()),
        EndingInString
    );
}

#[derive(Clone, Debug, Eq, PartialEq, Arbitrary, Dearbitrary)]
struct Empty;

#[test]
fn derive_empty() {
    assert_dearb_arb_eq!(Empty, Empty);
}

#[derive(Clone, Debug, Eq, PartialEq, Arbitrary, Dearbitrary)]
enum Color {
    A(Vec<u8>, bool),
    B,
    C { _x: u64, _v: Box<Color> },
    D(String),
}

#[test]
fn derive_enum() {
    // random lot of samples
    let v = vec![
        Color::B,
        Color::D("aaa".to_string()),
        Color::A(vec![1, 2, 3], false),
        Color::C {
            _x: 100,
            _v: Box::new(Color::B),
        },
        Color::C {
            _x: 100,
            _v: Box::new(Color::D("".to_string())),
        },
        Color::C {
            _x: 100,
            _v: Box::new(Color::A(vec![], false)),
        },
        Color::A(vec![], true),
        Color::C {
            _x: 100,
            _v: Box::new(Color::C {
                _x: 100,
                _v: Box::new(Color::A(vec![], false)),
            }),
        },
    ];
    for e in v.clone().into_iter() {
        assert_dearb_arb_eq!(e.clone(), Color);
    }
    for i in 0..=v.len() {
        assert_dearb_arb_eq!(v[..i].to_vec(), Vec<Color>);
    }
    for i in 0..=v.len() {
        assert_dearb_arb_eq!(v[i..].to_vec(), Vec<Color>);
    }
}
