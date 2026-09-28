//! Domains, read as the types they are over.
//!
//! A domain (`CREATE DOMAIN positive AS integer CHECK (VALUE > 0)`) adds constraints to a
//! base type without changing its representation: on the wire and in Arrow, a value of a
//! domain is a value of its base type. `FromSql` implementations accept only the types they
//! name, though, so without this a domain over `integer` is refused wherever an `integer`
//! would be read.

use std::error::Error;
use tokio_postgres::types::{FromSql, Kind, Type};

/// The type `ty` is ultimately over, following domains of domains; `ty` itself if it is
/// not a domain.
pub(crate) fn base(mut ty: &Type) -> &Type {
    while let Kind::Domain(inner) = ty.kind() {
        ty = inner;
    }
    ty
}

/// `ty` as the decoder dispatches on it: a domain as its base, and an array of domains as
/// the built-in array of their base, where the decoder knows one.
///
/// An array of a domain has a type of its own (`_positive`, not `_int4`), which no
/// built-in array matches by identity. Its elements read as their base all the same, so
/// only the type dispatched on needs to change.
pub(crate) fn normalized(ty: &Type) -> Type {
    let ty = base(ty);
    if let Kind::Array(element) = ty.kind() {
        let element_base = base(element);
        if element_base != element {
            if let Some(array) = array_of(element_base) {
                return array;
            }
        }
    }
    ty.clone()
}

/// The built-in array of `element`, for the element types the decoder reads arrays of.
fn array_of(element: &Type) -> Option<Type> {
    Some(match *element {
        Type::INT2 => Type::INT2_ARRAY,
        Type::INT4 => Type::INT4_ARRAY,
        Type::INT8 => Type::INT8_ARRAY,
        Type::OID => Type::OID_ARRAY,
        Type::FLOAT4 => Type::FLOAT4_ARRAY,
        Type::FLOAT8 => Type::FLOAT8_ARRAY,
        Type::TEXT => Type::TEXT_ARRAY,
        Type::BOOL => Type::BOOL_ARRAY,
        Type::BYTEA => Type::BYTEA_ARRAY,
        Type::NUMERIC => Type::NUMERIC_ARRAY,
        _ => return None,
    })
}

/// Reads a value of a domain as a value of its base type, and any other value as it is.
pub(crate) struct Transparent<T>(pub(crate) T);

impl<'a, T: FromSql<'a>> FromSql<'a> for Transparent<T> {
    fn from_sql(ty: &Type, raw: &'a [u8]) -> Result<Self, Box<dyn Error + Sync + Send>> {
        T::from_sql(base(ty), raw).map(Self)
    }

    fn from_sql_null(ty: &Type) -> Result<Self, Box<dyn Error + Sync + Send>> {
        T::from_sql_null(base(ty)).map(Self)
    }

    fn from_sql_nullable(
        ty: &Type,
        raw: Option<&'a [u8]>,
    ) -> Result<Self, Box<dyn Error + Sync + Send>> {
        T::from_sql_nullable(base(ty), raw).map(Self)
    }

    fn accepts(ty: &Type) -> bool {
        T::accepts(base(ty))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn domain(name: &str, over: Type) -> Type {
        Type::new(name.into(), 0, Kind::Domain(over), "public".into())
    }

    #[test]
    fn a_domain_resolves_to_its_base_through_other_domains() {
        let positive = domain("positive", Type::INT4);
        let small = domain("small", positive.clone());
        assert_eq!(base(&small), &Type::INT4);
        assert_eq!(base(&positive), &Type::INT4);
        assert_eq!(base(&Type::TEXT), &Type::TEXT);
    }

    #[test]
    fn a_domain_is_read_as_its_base_type() {
        let small = domain("small", domain("positive", Type::INT4));
        assert!(
            !<i32 as FromSql>::accepts(&small),
            "the plain reader refuses it"
        );
        assert!(<Transparent<i32> as FromSql>::accepts(&small));
        assert!(<Transparent<Option<i32>> as FromSql>::accepts(&small));
        assert!(!<Transparent<String> as FromSql>::accepts(&small));

        let value =
            Transparent::<Option<i32>>::from_sql_nullable(&small, Some(&7i32.to_be_bytes()))
                .unwrap();
        assert_eq!(value.0, Some(7));
        let null = Transparent::<Option<i32>>::from_sql_nullable(&small, None).unwrap();
        assert_eq!(null.0, None);
    }

    #[test]
    fn an_array_of_domains_dispatches_as_the_array_of_their_base() {
        let positive = domain("positive", Type::INT4);
        let array = Type::new(
            "_positive".into(),
            0,
            Kind::Array(positive.clone()),
            "public".into(),
        );
        assert_eq!(normalized(&array), Type::INT4_ARRAY);
        assert_eq!(normalized(&positive), Type::INT4);
        assert_eq!(normalized(&Type::TEXT_ARRAY), Type::TEXT_ARRAY);
    }

    #[test]
    fn an_array_of_domains_is_read_element_by_element() {
        let positive = domain("positive", Type::INT4);
        let array = Type::new(
            "_positive".into(),
            0,
            Kind::Array(positive),
            "public".into(),
        );
        assert!(!<Vec<i32> as FromSql>::accepts(&array));
        assert!(<Vec<Transparent<i32>> as FromSql>::accepts(&array));
    }
}
