//! Composite and array values, decoded recursively.
//!
//! A composite may hold composites and arrays, and an array composites, to any depth: a list
//! of structs whose members are themselves lists of structs, say. Each value is decoded from
//! its raw bytes by its Postgres type, into the builder its Arrow type produces. Both come
//! from here — [`data_type_with`] for the schema, [`append`] for the rows — so the two
//! cannot drift apart.
//!
//! A `Type` says what a composite holds but not its attributes' type modifiers, which is
//! where a `numeric`'s precision and scale live. The schema reads those from the catalog as
//! [`Modifiers`]; the rows are decoded into whatever Arrow type the schema settled on.
//!
//! Builders are Arrow's own [`make_builder`]: a `StructBuilder` for a struct and a
//! `ListBuilder<Box<dyn ArrayBuilder>>` for a list, at every depth.

use super::composite::CompositeType;
use super::domain;
use super::schema::{DEFAULT_NUMERIC_PRECISION, DEFAULT_NUMERIC_SCALE};
use super::{
    decimal_to_i128_mantissa, EnumValueFromSql, FailedToDecodeNestedValueSnafu,
    FailedToDowncastBuilderSnafu, FailedToGetCompositeRowValueSnafu, GeometryFromSql,
    JsonbRawString, Result, UnsupportedDataTypeSnafu,
};
use arrow::array::{
    ArrayBuilder, BinaryBuilder, BooleanBuilder, Date32Builder, Decimal128Builder, Float32Builder,
    Float64Builder, Int16Builder, Int32Builder, Int64Builder, ListBuilder, StringBuilder,
    StringDictionaryBuilder, StructBuilder, Time64NanosecondBuilder, TimestampNanosecondBuilder,
    UInt32Builder,
};
use arrow::datatypes::{DataType, Date32Type, Field, Fields, Int8Type, TimeUnit};
use chrono::{DateTime, NaiveDate, NaiveDateTime, NaiveTime, Timelike, Utc};
use rust_decimal::Decimal;
use snafu::prelude::*;
use snafu::IntoError;
use std::collections::HashMap;
use std::error::Error as StdError;
use std::sync::Arc;
use tokio_postgres::types::{FromSql, Kind, Type};
use tokio_postgres::Row;

pub(crate) use arrow::array::make_builder;

/// Whether a column of type `ty` is decoded here, rather than by the arms `rows_to_arrow`
/// has for scalars and for arrays of the common scalars: every composite, and every array
/// no such arm handles.
pub(crate) fn decodes(ty: &Type) -> bool {
    let ty = domain::normalized(ty);
    match ty.kind() {
        Kind::Composite(_) => true,
        Kind::Array(_) => {
            !matches!(
                ty,
                Type::INT2_ARRAY
                    | Type::INT4_ARRAY
                    | Type::INT8_ARRAY
                    | Type::OID_ARRAY
                    | Type::FLOAT4_ARRAY
                    | Type::FLOAT8_ARRAY
                    | Type::TEXT_ARRAY
                    | Type::BOOL_ARRAY
                    | Type::BYTEA_ARRAY
                    | Type::NUMERIC_ARRAY
            ) && !matches!(ty.name(), "_geometry" | "_geography")
        }
        _ => false,
    }
}

/// Each composite attribute's type modifier (`atttypmod`, through any domain), by the
/// composite's type OID and the attribute's position among its live attributes.
pub(crate) type Modifiers = HashMap<(u32, usize), i32>;

/// The Arrow type a value of `ty` is decoded into, not knowing any type modifiers: a
/// `numeric` anywhere in it takes the pinned precision and scale of an unconstrained one.
pub(crate) fn data_type(ty: &Type, field_name: &str) -> Result<DataType> {
    data_type_with(ty, -1, &Modifiers::new(), field_name)
}

/// The Arrow type a value of `ty`, declared with type modifier `modifier`, is decoded into,
/// with `modifiers` giving those of the attributes of any composite in it.
pub(crate) fn data_type_with(
    ty: &Type,
    modifier: i32,
    modifiers: &Modifiers,
    field_name: &str,
) -> Result<DataType> {
    let ty = domain::base(ty);
    Ok(match *ty {
        Type::BOOL => DataType::Boolean,
        Type::INT2 => DataType::Int16,
        Type::INT4 => DataType::Int32,
        Type::INT8 => DataType::Int64,
        Type::OID => DataType::UInt32,
        Type::FLOAT4 => DataType::Float32,
        Type::FLOAT8 => DataType::Float64,
        Type::TEXT
        | Type::VARCHAR
        | Type::BPCHAR
        | Type::NAME
        | Type::UUID
        | Type::JSON
        | Type::JSONB => DataType::Utf8,
        Type::BYTEA => DataType::Binary,
        Type::DATE => DataType::Date32,
        Type::TIME => DataType::Time64(TimeUnit::Nanosecond),
        Type::TIMESTAMP => DataType::Timestamp(TimeUnit::Nanosecond, None),
        Type::TIMESTAMPTZ => DataType::Timestamp(TimeUnit::Nanosecond, Some(Arc::from("UTC"))),
        Type::NUMERIC => numeric_type(modifier),
        _ if is_geometry(ty) => DataType::Binary,
        _ => match ty.kind() {
            Kind::Enum(_) => {
                DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8))
            }
            Kind::Composite(fields) => {
                let fields = fields
                    .iter()
                    .enumerate()
                    .map(|(index, field)| {
                        let modifier = modifiers.get(&(ty.oid(), index)).copied().unwrap_or(-1);
                        data_type_with(field.type_(), modifier, modifiers, field.name())
                            .map(|data_type| Field::new(field.name(), data_type, true))
                    })
                    .collect::<Result<Vec<_>>>()?;
                DataType::Struct(Fields::from(fields))
            }
            // An array's modifier is its element's.
            Kind::Array(element) => DataType::List(Arc::new(Field::new(
                "item",
                data_type_with(element, modifier, modifiers, field_name)?,
                true,
            ))),
            _ => {
                return UnsupportedDataTypeSnafu {
                    data_type: ty.to_string(),
                    field_name,
                }
                .fail()
            }
        },
    })
}

/// The Arrow type of a `numeric` declared with `modifier`: its own precision and scale if
/// it declares them and `Decimal128` can hold them, and the pinned ones otherwise.
fn numeric_type(modifier: i32) -> DataType {
    // Postgres packs `numeric(p,s)` as `((p << 16) | s) + VARHDRSZ`.
    if modifier >= 4 {
        let packed = modifier - 4;
        if let (Ok(precision), Ok(scale)) =
            (u8::try_from(packed >> 16), i8::try_from(packed & 0xffff))
        {
            if (1..=38).contains(&precision) && scale >= 0 {
                return DataType::Decimal128(precision, scale);
            }
        }
    }
    DataType::Decimal128(DEFAULT_NUMERIC_PRECISION, DEFAULT_NUMERIC_SCALE)
}

/// The OIDs of the composite types `ty` holds at any depth, itself included, added to
/// `oids`: the types whose [`Modifiers`] a schema needs.
pub(crate) fn composites(ty: &Type, oids: &mut Vec<u32>) {
    let ty = domain::base(ty);
    match ty.kind() {
        Kind::Composite(fields) => {
            if !oids.contains(&ty.oid()) {
                oids.push(ty.oid());
                for field in fields {
                    composites(field.type_(), oids);
                }
            }
        }
        Kind::Array(element) => composites(element, oids),
        _ => {}
    }
}

/// Column `i` of `row`, as the bytes it arrived as; `None` for null.
pub(crate) fn read(
    row: &Row,
    i: usize,
) -> std::result::Result<Option<&[u8]>, tokio_postgres::Error> {
    row.try_get::<usize, Raw>(i).map(|raw| raw.0)
}

/// Appends one value of `ty`, given as its raw bytes, to `builder`, which [`make_builder`]
/// made for `data_type`: the Arrow type the schema gives it.
pub(crate) fn append(
    builder: &mut dyn ArrayBuilder,
    ty: &Type,
    data_type: &DataType,
    raw: Option<&[u8]>,
    field_name: &str,
) -> Result<()> {
    let ty = domain::base(ty);
    let Some(raw) = raw else {
        return append_null(builder, ty, field_name);
    };

    match ty.kind() {
        Kind::Composite(fields) => {
            let DataType::Struct(arrow_fields) = data_type else {
                return mismatch(ty, data_type, field_name);
            };
            if arrow_fields.len() != fields.len() {
                return mismatch(ty, data_type, field_name);
            }
            let builder = downcast::<StructBuilder>(builder, ty)?;
            let composite = CompositeType::from_sql(ty, raw)
                .map_err(|source| decode_error(ty, field_name, source))?;
            for (index, field) in fields.iter().enumerate() {
                let value = composite.try_get::<usize, Raw>(index).context(
                    FailedToGetCompositeRowValueSnafu {
                        pg_type: ty.clone(),
                    },
                )?;
                append(
                    builder.field_builders_mut()[index].as_mut(),
                    field.type_(),
                    arrow_fields[index].data_type(),
                    value.0,
                    field.name(),
                )?;
            }
            builder.append(true);
        }
        Kind::Array(element) => {
            let DataType::List(item) = data_type else {
                return mismatch(ty, data_type, field_name);
            };
            let builder = downcast::<ListBuilder<Box<dyn ArrayBuilder>>>(builder, ty)?;
            let values = Vec::<Raw>::from_sql(ty, raw)
                .map_err(|source| decode_error(ty, field_name, source))?;
            for value in values {
                append(
                    builder.values().as_mut(),
                    element,
                    item.data_type(),
                    value.0,
                    field_name,
                )?;
            }
            builder.append(true);
        }
        _ => append_scalar(builder, ty, data_type, raw, field_name)?,
    }
    Ok(())
}

fn append_scalar(
    builder: &mut dyn ArrayBuilder,
    ty: &Type,
    data_type: &DataType,
    raw: &[u8],
    field_name: &str,
) -> Result<()> {
    /// Decodes the value as `$Value` and appends `$convert` of it to a `$Builder`.
    macro_rules! put {
        ($Builder:ty, $Value:ty, |$v:ident| $convert:expr) => {{
            let $v = <$Value as FromSql>::from_sql(ty, raw)
                .map_err(|source| decode_error(ty, field_name, source))?;
            let value = $convert;
            downcast::<$Builder>(builder, ty)?.append_value(value);
        }};
    }

    // The same conversions `rows_to_arrow` makes for a column of each type.
    match *ty {
        Type::BOOL => put!(BooleanBuilder, bool, |v| v),
        Type::INT2 => put!(Int16Builder, i16, |v| v),
        Type::INT4 => put!(Int32Builder, i32, |v| v),
        Type::INT8 => put!(Int64Builder, i64, |v| v),
        Type::OID => put!(UInt32Builder, u32, |v| v),
        Type::FLOAT4 => put!(Float32Builder, f32, |v| v),
        Type::FLOAT8 => put!(Float64Builder, f64, |v| v),
        Type::TEXT | Type::VARCHAR | Type::BPCHAR | Type::NAME => {
            put!(StringBuilder, &str, |v| v)
        }
        Type::UUID => put!(StringBuilder, uuid::Uuid, |v| v.to_string()),
        Type::JSON | Type::JSONB => put!(StringBuilder, JsonbRawString, |v| v.0),
        Type::BYTEA => put!(BinaryBuilder, &[u8], |v| v),
        Type::DATE => put!(Date32Builder, NaiveDate, |v| Date32Type::from_naive_date(v)),
        Type::TIME => put!(Time64NanosecondBuilder, NaiveTime, |v| i64::from(
            v.num_seconds_from_midnight()
        ) * 1_000_000_000
            + i64::from(v.nanosecond())),
        Type::TIMESTAMP => put!(TimestampNanosecondBuilder, NaiveDateTime, |v| nanos(
            v.and_utc(),
            ty,
            field_name
        )?),
        Type::TIMESTAMPTZ => put!(TimestampNanosecondBuilder, DateTime<Utc>, |v| nanos(
            v, ty, field_name
        )?),
        Type::NUMERIC => {
            let DataType::Decimal128(_, scale) = data_type else {
                return mismatch(ty, data_type, field_name);
            };
            let scale = u32::try_from(*scale).unwrap_or_default();
            put!(Decimal128Builder, Decimal, |v| decimal_to_i128_mantissa(
                v, scale, field_name
            )?)
        }
        _ if is_geometry(ty) => put!(BinaryBuilder, GeometryFromSql, |v| v.wkb),
        _ if matches!(ty.kind(), Kind::Enum(_)) => {
            put!(StringDictionaryBuilder<Int8Type>, EnumValueFromSql, |v| v
                .enum_value)
        }
        _ => {
            return UnsupportedDataTypeSnafu {
                data_type: ty.to_string(),
                field_name,
            }
            .fail()
        }
    }
    Ok(())
}

/// Appends a null to `builder`. A struct's fields each take a null as well, so they stay as
/// long as the struct.
fn append_null(builder: &mut dyn ArrayBuilder, ty: &Type, field_name: &str) -> Result<()> {
    let builder = builder.as_any_mut();
    if let Some(builder) = builder.downcast_mut::<StructBuilder>() {
        // A null struct says nothing about its fields' types, so each is nulled by the
        // builder it has.
        for field in builder.field_builders_mut() {
            append_null(field.as_mut(), ty, field_name)?;
        }
        builder.append_null();
        return Ok(());
    }

    macro_rules! null {
        ($($Builder:ty),*) => {
            $(
                if let Some(builder) = builder.downcast_mut::<$Builder>() {
                    builder.append_null();
                    return Ok(());
                }
            )*
        };
    }
    null!(
        ListBuilder<Box<dyn ArrayBuilder>>,
        BooleanBuilder,
        Int16Builder,
        Int32Builder,
        Int64Builder,
        UInt32Builder,
        Float32Builder,
        Float64Builder,
        StringBuilder,
        BinaryBuilder,
        Date32Builder,
        Time64NanosecondBuilder,
        TimestampNanosecondBuilder,
        Decimal128Builder,
        StringDictionaryBuilder<Int8Type>
    );

    FailedToDowncastBuilderSnafu {
        postgres_type: format!("{ty} (a null in `{field_name}`)"),
    }
    .fail()
}

fn downcast<'b, B: ArrayBuilder>(
    builder: &'b mut dyn ArrayBuilder,
    ty: &Type,
) -> Result<&'b mut B> {
    builder
        .as_any_mut()
        .downcast_mut::<B>()
        .context(FailedToDowncastBuilderSnafu {
            postgres_type: ty.to_string(),
        })
}

/// `data_type` is not the shape a value of `ty` decodes into.
fn mismatch<T>(ty: &Type, data_type: &DataType, field_name: &str) -> Result<T> {
    Err(decode_error(
        ty,
        field_name,
        format!("cannot decode it as {data_type}").into(),
    ))
}

fn nanos(value: DateTime<Utc>, ty: &Type, field_name: &str) -> Result<i64> {
    value.timestamp_nanos_opt().ok_or_else(|| {
        decode_error(
            ty,
            field_name,
            "out of range for a nanosecond timestamp".into(),
        )
    })
}

fn is_geometry(ty: &Type) -> bool {
    matches!(ty.name(), "geometry" | "geography")
}

fn decode_error(
    ty: &Type,
    field_name: &str,
    source: Box<dyn StdError + Sync + Send>,
) -> super::Error {
    FailedToDecodeNestedValueSnafu {
        pg_type: ty.clone(),
        field_name,
    }
    .into_error(source)
}

/// A value's bytes as they arrived, whatever its type; `None` for null.
struct Raw<'a>(Option<&'a [u8]>);

impl<'a> FromSql<'a> for Raw<'a> {
    fn from_sql(
        _: &Type,
        raw: &'a [u8],
    ) -> std::result::Result<Self, Box<dyn StdError + Sync + Send>> {
        Ok(Raw(Some(raw)))
    }

    fn from_sql_null(_: &Type) -> std::result::Result<Self, Box<dyn StdError + Sync + Send>> {
        Ok(Raw(None))
    }

    fn accepts(_: &Type) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_postgres::types::Field as PgField;

    fn composite(name: &str, oid: u32, fields: Vec<(&str, Type)>) -> Type {
        let fields = fields
            .into_iter()
            .map(|(name, ty)| PgField::new(name.into(), ty))
            .collect();
        Type::new(name.into(), oid, Kind::Composite(fields), "public".into())
    }

    fn array_of(element: Type) -> Type {
        Type::new(
            format!("_{}", element.name()),
            0,
            Kind::Array(element),
            "public".into(),
        )
    }

    /// `numeric(p,s)` as Postgres packs it.
    fn numeric(precision: i32, scale: i32) -> i32 {
        ((precision << 16) | scale) + 4
    }

    #[test]
    fn a_numeric_keeps_what_it_declares_when_decimal128_can_hold_it() {
        assert_eq!(numeric_type(numeric(10, 2)), DataType::Decimal128(10, 2));
        let pinned = DataType::Decimal128(DEFAULT_NUMERIC_PRECISION, DEFAULT_NUMERIC_SCALE);
        assert_eq!(numeric_type(-1), pinned, "unconstrained");
        assert_eq!(
            numeric_type(numeric(50, 2)),
            pinned,
            "wider than Decimal128"
        );
    }

    #[test]
    fn composites_and_uncommon_arrays_are_decoded_here() {
        let tag = composite("tag", 1, vec![("code", Type::TEXT)]);
        assert!(decodes(&tag));
        assert!(decodes(&array_of(tag)));
        assert!(decodes(&Type::VARCHAR_ARRAY));
        assert!(decodes(&Type::TIMESTAMPTZ_ARRAY));
        assert!(
            !decodes(&Type::INT4_ARRAY),
            "the common arrays have arms of their own"
        );
        assert!(!decodes(&Type::TEXT));
    }

    #[test]
    fn modifiers_reach_numerics_at_any_depth() {
        let inner = composite("inner", 2, vec![("amount", Type::NUMERIC)]);
        let outer = composite(
            "outer",
            1,
            vec![("n", Type::NUMERIC), ("inners", array_of(inner))],
        );
        let modifiers = Modifiers::from([((1, 0), numeric(5, 1)), ((2, 0), numeric(10, 2))]);

        let DataType::Struct(fields) = data_type_with(&outer, -1, &modifiers, "c").unwrap() else {
            panic!("a composite is a struct");
        };
        assert_eq!(fields[0].data_type(), &DataType::Decimal128(5, 1));
        let DataType::List(item) = fields[1].data_type() else {
            panic!("an array is a list");
        };
        let DataType::Struct(inner) = item.data_type() else {
            panic!("of structs");
        };
        assert_eq!(inner[0].data_type(), &DataType::Decimal128(10, 2));

        let mut oids = Vec::new();
        composites(&outer, &mut oids);
        assert_eq!(oids, [1, 2]);
    }
}
