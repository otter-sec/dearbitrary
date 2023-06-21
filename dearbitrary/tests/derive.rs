#![cfg(feature = "derive")]
// Various structs/fields that we are deriving `Arbitrary` for aren't actually
// used except to exercise the derive.
#![allow(dead_code)]

use arbitrary::*;
use dearbitrary::*;

macro_rules! assert_dearb_arb_eq {
    ($v:expr, $t:ty) => {{
        // FIXME: dearbitrary first does now work for now

        // with take rest
        // let x: $t = $v;
        // let bytes = x.dearbitrary_first().finish();
        //
        // let mut u = Unstructured::new(&bytes);
        // let y = <$t>::arbitrary_take_rest(u).unwrap();
        // assert_eq!(x, y);

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
    // let bytes = [1, 1, 1, 2, 3, 4, 5, 6, 7, 8];
    // let s1 = EndingInVec::arbitrary_take_rest(Unstructured::new(&bytes)).unwrap();
    // let s2 = EndingInString::arbitrary_take_rest(Unstructured::new(&bytes)).unwrap();
    // assert_eq!(s1.0, 1);
    // assert_eq!(s2.0, 1);
    // assert_eq!(s1.1, true);
    // assert_eq!(s2.1, true);
    // assert_eq!(s1.2, 0x4030201);
    // assert_eq!(s2.2, 0x4030201);
    // assert_eq!(s1.3, vec![0x605, 0x807]);
    // assert_eq!(s2.3, "\x05\x06\x07\x08");
    // assert_dearb_arb_eq!(s1);
    // assert_dearb_arb_eq!(s2);
    assert_dearb_arb_eq!(
        EndingInVec(1, false, 0x4030201, vec![0x605, 0x807]),
        EndingInVec
    );
    assert_dearb_arb_eq!(
        EndingInString(1, false, 0x4030201, "\x05\x06\x07\x08".to_string()),
        EndingInString
    );
}
