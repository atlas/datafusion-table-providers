//! Arrow record batches written with PostgreSQL's binary `COPY`.
//!
//! Every value is encoded in the binary form of the column it is written to, as the
//! server's own receive function for that type reads it, so no value passes through SQL
//! text. The column's type, not the Arrow type, decides the encoding: domains are written
//! as the type they are over, composites field by field and arrays element by element,
//! each in its own type's form. A dictionary column is written as its values. [`supported`]
//! says whether every column of a batch can be written this way.

use std::pin::pin;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, RecordBatch, StructArray};
use arrow::compute::cast;
use arrow::datatypes::{
    DataType, Date32Type, Date64Type, DurationMicrosecondType, DurationMillisecondType,
    DurationNanosecondType, DurationSecondType, Fields, Float16Type, Float32Type, Float64Type,
    Int16Type, Int32Type, Int64Type, Int8Type, IntervalDayTimeType, IntervalMonthDayNanoType,
    IntervalUnit, IntervalYearMonthType, Time32MillisecondType, Time32SecondType,
    Time64MicrosecondType, Time64NanosecondType, TimeUnit, TimestampMicrosecondType,
    TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType, UInt16Type, UInt32Type,
    UInt64Type, UInt8Type,
};
use arrow::error::ArrowError;
use arrow::util::display::array_value_to_string;
use bytes::{BufMut, BytesMut};
use snafu::prelude::*;
use tokio_postgres::binary_copy::BinaryCopyInWriter;
use tokio_postgres::types::{to_sql_checked, IsNull, Kind, ToSql, Type};
use tokio_postgres::Transaction;

type BoxError = Box<dyn std::error::Error + Sync + Send>;

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("{source}"))]
    Database { source: tokio_postgres::Error },
    #[snafu(display("Unable to read a dictionary column's values: {source}"))]
    Dictionary { source: ArrowError },
}

/// Days from 1970-01-01, Arrow's epoch, to 2000-01-01, PostgreSQL's.
const EPOCH_DAYS: i64 = 10_957;
/// Microseconds from 1970-01-01 to 2000-01-01.
const EPOCH_MICROS: i64 = EPOCH_DAYS * 86_400_000_000;
/// The version byte that starts a `jsonb` value.
const JSONB_VERSION: u8 = 1;
/// The sign of a negative `numeric`.
const NUMERIC_NEGATIVE: u16 = 0x4000;

/// Whether values of the Arrow types in `fields` can each be written to the column of
/// the same position in `types`.
pub fn supported(fields: &Fields, types: &[Type]) -> bool {
    fields.len() == types.len()
        && fields
            .iter()
            .zip(types)
            .all(|(field, ty)| match field.data_type() {
                DataType::Dictionary(_, values) => supports(values, ty),
                arrow => supports(arrow, ty),
            })
}

fn supports(arrow: &DataType, ty: &Type) -> bool {
    if arrow == &DataType::Null {
        return true;
    }
    match ty.kind() {
        Kind::Domain(base) => supports(arrow, base),
        Kind::Array(element) => match arrow {
            DataType::List(item) | DataType::LargeList(item) | DataType::FixedSizeList(item, _) => {
                // An array of arrays is one multidimensional array, which must be
                // rectangular; lists of lists need not be.
                !matches!(
                    item.data_type(),
                    DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(_, _)
                ) && supports(item.data_type(), element)
            }
            _ => false,
        },
        Kind::Composite(members) => match arrow {
            DataType::Struct(fields) => {
                fields.len() == members.len()
                    && members.iter().all(|member| {
                        fields
                            .find(member.name())
                            .is_some_and(|(_, field)| supports(field.data_type(), member.type_()))
                    })
            }
            _ => false,
        },
        Kind::Enum(_) => is_text(arrow),
        Kind::Simple => supports_simple(arrow, ty),
        _ => false,
    }
}

fn supports_simple(arrow: &DataType, ty: &Type) -> bool {
    match *ty {
        Type::BOOL => arrow == &DataType::Boolean,
        Type::INT2 | Type::INT4 => arrow.is_integer(),
        Type::INT8 => arrow.is_integer() || matches!(arrow, DataType::Duration(_)),
        Type::FLOAT4 => matches!(arrow, DataType::Float16 | DataType::Float32),
        Type::FLOAT8 => matches!(
            arrow,
            DataType::Float16 | DataType::Float32 | DataType::Float64
        ),
        Type::NUMERIC => {
            arrow.is_integer()
                || matches!(
                    arrow,
                    DataType::Decimal32(_, _)
                        | DataType::Decimal64(_, _)
                        | DataType::Decimal128(_, _)
                        | DataType::Decimal256(_, _)
                )
        }
        Type::TEXT | Type::VARCHAR | Type::BPCHAR | Type::NAME | Type::JSON | Type::JSONB => {
            is_text(arrow)
        }
        Type::UUID => is_text(arrow) || arrow == &DataType::FixedSizeBinary(16),
        Type::BYTEA => is_binary(arrow),
        Type::DATE => matches!(arrow, DataType::Date32 | DataType::Date64),
        Type::TIME => matches!(arrow, DataType::Time32(_) | DataType::Time64(_)),
        Type::TIMESTAMP | Type::TIMESTAMPTZ => matches!(arrow, DataType::Timestamp(_, _)),
        Type::INTERVAL => matches!(arrow, DataType::Interval(_)),
        // PostGIS reads its binary form as WKB or EWKB.
        _ if matches!(ty.name(), "geometry" | "geography") => is_binary(arrow),
        _ if ty.name() == "citext" => is_text(arrow),
        _ => false,
    }
}

fn is_text(arrow: &DataType) -> bool {
    matches!(
        arrow,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
    )
}

fn is_binary(arrow: &DataType) -> bool {
    matches!(
        arrow,
        DataType::Binary
            | DataType::LargeBinary
            | DataType::BinaryView
            | DataType::FixedSizeBinary(_)
    )
}

/// Writes `batch` to `columns` of `table`, both already quoted, whose types are `types`
/// in the same order, and returns how many rows were written. Every column must be
/// [`supported`].
pub async fn copy_in(
    transaction: &Transaction<'_>,
    table: &str,
    columns: &str,
    batch: &RecordBatch,
    types: &[Type],
) -> Result<u64, Error> {
    let sink = transaction
        .copy_in(&format!(
            "COPY {table} ({columns}) FROM STDIN (FORMAT binary)"
        ))
        .await
        .context(DatabaseSnafu)?;
    let arrays = batch
        .columns()
        .iter()
        .map(|array| match array.data_type() {
            DataType::Dictionary(_, values) => cast(array, values),
            _ => Ok(Arc::clone(array)),
        })
        .collect::<Result<Vec<_>, _>>()
        .context(DictionarySnafu)?;
    let mut writer = pin!(BinaryCopyInWriter::new(sink, types));
    for row in 0..batch.num_rows() {
        let cells: Vec<Cell> = arrays.iter().map(|array| Cell { array, row }).collect();
        let values: Vec<&(dyn ToSql + Sync)> = cells.iter().map(|cell| cell as _).collect();
        writer
            .as_mut()
            .write_raw(values)
            .await
            .context(DatabaseSnafu)?;
    }
    writer.finish().await.context(DatabaseSnafu)
}

/// The value in `row` of `array`, written in the binary form of the type it lands in.
#[derive(Debug)]
struct Cell<'a> {
    array: &'a ArrayRef,
    row: usize,
}

impl ToSql for Cell<'_> {
    fn to_sql(&self, ty: &Type, out: &mut BytesMut) -> Result<IsNull, BoxError> {
        if is_null(self.array.as_ref(), self.row) {
            return Ok(IsNull::Yes);
        }
        encode(self.array.as_ref(), self.row, ty, out)?;
        Ok(IsNull::No)
    }

    // Whether a column can be written is decided for the whole batch, by `supported`.
    fn accepts(_: &Type) -> bool {
        true
    }

    to_sql_checked!();
}

/// Whether the value in `row` of `array` is null, including in an array of type `Null`.
fn is_null(array: &dyn Array, row: usize) -> bool {
    array
        .logical_nulls()
        .is_some_and(|nulls| nulls.is_null(row))
}

/// Appends the value in `row` of `array`, which is not null, in the binary form of `ty`.
fn encode(array: &dyn Array, row: usize, ty: &Type, out: &mut BytesMut) -> Result<(), BoxError> {
    match ty.kind() {
        Kind::Domain(base) => encode(array, row, base, out),
        Kind::Array(element) => encode_array(array, row, element, out),
        Kind::Composite(members) => {
            let composite = array.as_struct();
            out.put_i32(i32::try_from(members.len())?);
            for member in members {
                let field = field(composite, member.name())?;
                out.put_u32(member.type_().oid());
                encode_prefixed(field.as_ref(), row, member.type_(), out)?;
            }
            Ok(())
        }
        Kind::Enum(_) => {
            out.put_slice(text(array, row)?.as_bytes());
            Ok(())
        }
        _ => encode_simple(array, row, ty, out),
    }
}

fn field<'a>(composite: &'a StructArray, name: &str) -> Result<&'a ArrayRef, BoxError> {
    composite
        .column_by_name(name)
        .ok_or_else(|| format!("no field {name} for the composite's member of that name").into())
}

/// Appends the value in `row` of `array` as its length and then its binary form, or as
/// a length of -1 when it is null.
fn encode_prefixed(
    array: &dyn Array,
    row: usize,
    ty: &Type,
    out: &mut BytesMut,
) -> Result<(), BoxError> {
    if is_null(array, row) {
        out.put_i32(-1);
        return Ok(());
    }
    let start = out.len();
    out.put_i32(0);
    encode(array, row, ty, out)?;
    let length = i32::try_from(out.len() - start - 4)?;
    out[start..start + 4].copy_from_slice(&length.to_be_bytes());
    Ok(())
}

/// Appends the list in `row` of `array` as a one-dimensional array of `element`, or as
/// an empty array.
fn encode_array(
    array: &dyn Array,
    row: usize,
    element: &Type,
    out: &mut BytesMut,
) -> Result<(), BoxError> {
    let items = match array.data_type() {
        DataType::List(_) => array.as_list::<i32>().value(row),
        DataType::LargeList(_) => array.as_list::<i64>().value(row),
        DataType::FixedSizeList(_, _) => array.as_fixed_size_list().value(row),
        other => return Err(format!("{other} cannot be written as an array").into()),
    };
    let has_null = (0..items.len()).any(|item| is_null(items.as_ref(), item));
    out.put_i32(i32::from(!items.is_empty()));
    out.put_i32(i32::from(has_null));
    out.put_u32(element.oid());
    if !items.is_empty() {
        out.put_i32(i32::try_from(items.len())?);
        out.put_i32(1);
    }
    for item in 0..items.len() {
        encode_prefixed(items.as_ref(), item, element, out)?;
    }
    Ok(())
}

fn encode_simple(
    array: &dyn Array,
    row: usize,
    ty: &Type,
    out: &mut BytesMut,
) -> Result<(), BoxError> {
    match *ty {
        Type::BOOL => out.put_u8(u8::from(array.as_boolean().value(row))),
        Type::INT2 => {
            out.put_i16(i16::try_from(integer(array, row)?).map_err(|_| out_of_range(ty))?)
        }
        Type::INT4 => {
            out.put_i32(i32::try_from(integer(array, row)?).map_err(|_| out_of_range(ty))?)
        }
        Type::INT8 => {
            out.put_i64(i64::try_from(integer(array, row)?).map_err(|_| out_of_range(ty))?)
        }
        Type::FLOAT4 => out.put_f32(match array.data_type() {
            DataType::Float16 => array.as_primitive::<Float16Type>().value(row).to_f32(),
            _ => array.as_primitive::<Float32Type>().value(row),
        }),
        Type::FLOAT8 => out.put_f64(match array.data_type() {
            DataType::Float16 => array.as_primitive::<Float16Type>().value(row).to_f64(),
            DataType::Float32 => f64::from(array.as_primitive::<Float32Type>().value(row)),
            _ => array.as_primitive::<Float64Type>().value(row),
        }),
        Type::NUMERIC => encode_numeric(&array_value_to_string(array, row)?, out)?,
        Type::JSONB => {
            out.put_u8(JSONB_VERSION);
            out.put_slice(text(array, row)?.as_bytes());
        }
        Type::UUID => match array.data_type() {
            DataType::FixedSizeBinary(16) => {
                out.put_slice(array.as_fixed_size_binary().value(row));
            }
            _ => out.put_slice(uuid::Uuid::parse_str(text(array, row)?)?.as_bytes()),
        },
        Type::BYTEA => out.put_slice(binary(array, row)?),
        Type::DATE => {
            let days = match array.data_type() {
                DataType::Date32 => i64::from(array.as_primitive::<Date32Type>().value(row)),
                _ => array
                    .as_primitive::<Date64Type>()
                    .value(row)
                    .div_euclid(86_400_000),
            };
            out.put_i32(i32::try_from(days - EPOCH_DAYS).map_err(|_| out_of_range(ty))?);
        }
        Type::TIME => out.put_i64(match array.data_type() {
            DataType::Time32(TimeUnit::Second) => {
                i64::from(array.as_primitive::<Time32SecondType>().value(row)) * 1_000_000
            }
            DataType::Time32(_) => {
                i64::from(array.as_primitive::<Time32MillisecondType>().value(row)) * 1_000
            }
            DataType::Time64(TimeUnit::Microsecond) => {
                array.as_primitive::<Time64MicrosecondType>().value(row)
            }
            _ => nanos_to_micros(array.as_primitive::<Time64NanosecondType>().value(row)),
        }),
        Type::TIMESTAMP | Type::TIMESTAMPTZ => {
            let micros = timestamp_micros(array, row).ok_or_else(|| out_of_range(ty))?;
            out.put_i64(
                micros
                    .checked_sub(EPOCH_MICROS)
                    .ok_or_else(|| out_of_range(ty))?,
            );
        }
        Type::INTERVAL => {
            let (months, days, micros) = match array.data_type() {
                DataType::Interval(IntervalUnit::MonthDayNano) => {
                    let value = array.as_primitive::<IntervalMonthDayNanoType>().value(row);
                    (value.months, value.days, nanos_to_micros(value.nanoseconds))
                }
                DataType::Interval(IntervalUnit::DayTime) => {
                    let value = array.as_primitive::<IntervalDayTimeType>().value(row);
                    (0, value.days, i64::from(value.milliseconds) * 1_000)
                }
                _ => (
                    array.as_primitive::<IntervalYearMonthType>().value(row),
                    0,
                    0,
                ),
            };
            out.put_i64(micros);
            out.put_i32(days);
            out.put_i32(months);
        }
        // `text`, `varchar`, `bpchar`, `name`, `json` and `citext` are read as their text.
        _ if is_text(array.data_type()) => out.put_slice(text(array, row)?.as_bytes()),
        // `geometry` and `geography`.
        _ => out.put_slice(binary(array, row)?),
    }
    Ok(())
}

fn out_of_range(ty: &Type) -> BoxError {
    format!("value out of range for type {}", ty.name()).into()
}

fn text(array: &dyn Array, row: usize) -> Result<&str, BoxError> {
    Ok(match array.data_type() {
        DataType::Utf8 => array.as_string::<i32>().value(row),
        DataType::LargeUtf8 => array.as_string::<i64>().value(row),
        DataType::Utf8View => array.as_string_view().value(row),
        other => return Err(format!("{other} cannot be written as text").into()),
    })
}

fn binary(array: &dyn Array, row: usize) -> Result<&[u8], BoxError> {
    Ok(match array.data_type() {
        DataType::Binary => array.as_binary::<i32>().value(row),
        DataType::LargeBinary => array.as_binary::<i64>().value(row),
        DataType::BinaryView => array.as_binary_view().value(row),
        DataType::FixedSizeBinary(_) => array.as_fixed_size_binary().value(row),
        other => return Err(format!("{other} cannot be written as bytes").into()),
    })
}

fn integer(array: &dyn Array, row: usize) -> Result<i128, BoxError> {
    Ok(match array.data_type() {
        DataType::Int8 => array.as_primitive::<Int8Type>().value(row).into(),
        DataType::Int16 => array.as_primitive::<Int16Type>().value(row).into(),
        DataType::Int32 => array.as_primitive::<Int32Type>().value(row).into(),
        DataType::Int64 => array.as_primitive::<Int64Type>().value(row).into(),
        DataType::UInt8 => array.as_primitive::<UInt8Type>().value(row).into(),
        DataType::UInt16 => array.as_primitive::<UInt16Type>().value(row).into(),
        DataType::UInt32 => array.as_primitive::<UInt32Type>().value(row).into(),
        DataType::UInt64 => array.as_primitive::<UInt64Type>().value(row).into(),
        DataType::Duration(TimeUnit::Second) => {
            array.as_primitive::<DurationSecondType>().value(row).into()
        }
        DataType::Duration(TimeUnit::Millisecond) => array
            .as_primitive::<DurationMillisecondType>()
            .value(row)
            .into(),
        DataType::Duration(TimeUnit::Microsecond) => array
            .as_primitive::<DurationMicrosecondType>()
            .value(row)
            .into(),
        DataType::Duration(TimeUnit::Nanosecond) => array
            .as_primitive::<DurationNanosecondType>()
            .value(row)
            .into(),
        other => return Err(format!("{other} cannot be written as an integer").into()),
    })
}

/// The timestamp in `row` of `array` in microseconds since 1970, or `None` if it has no
/// such value.
fn timestamp_micros(array: &dyn Array, row: usize) -> Option<i64> {
    match array.data_type() {
        DataType::Timestamp(TimeUnit::Second, _) => array
            .as_primitive::<TimestampSecondType>()
            .value(row)
            .checked_mul(1_000_000),
        DataType::Timestamp(TimeUnit::Millisecond, _) => array
            .as_primitive::<TimestampMillisecondType>()
            .value(row)
            .checked_mul(1_000),
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            Some(array.as_primitive::<TimestampMicrosecondType>().value(row))
        }
        _ => Some(nanos_to_micros(
            array.as_primitive::<TimestampNanosecondType>().value(row),
        )),
    }
}

/// Nanoseconds as microseconds, rounded half to even as PostgreSQL rounds a time it reads
/// with more precision than it keeps.
fn nanos_to_micros(nanos: i64) -> i64 {
    let (micros, remainder) = (nanos.div_euclid(1_000), nanos.rem_euclid(1_000));
    match remainder.cmp(&500) {
        std::cmp::Ordering::Less => micros,
        std::cmp::Ordering::Greater => micros + 1,
        std::cmp::Ordering::Equal => micros + (micros & 1),
    }
}

/// Appends `decimal`, a number such as `-12.3400`, as a `numeric`: base-10000 digits
/// around the decimal point, keeping every fractional digit written as its scale.
fn encode_numeric(decimal: &str, out: &mut BytesMut) -> Result<(), BoxError> {
    let invalid = || -> BoxError { format!("{decimal} is not a decimal number").into() };
    let (negative, unsigned) = match decimal.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, decimal),
    };
    let (whole, fraction) = unsigned.split_once('.').unwrap_or((unsigned, ""));
    if whole.is_empty() && fraction.is_empty()
        || !whole
            .bytes()
            .chain(fraction.bytes())
            .all(|b| b.is_ascii_digit())
    {
        return Err(invalid());
    }
    let scale = u16::try_from(fraction.len()).map_err(|_| invalid())?;
    // Groups of four digits, the whole part padded on the left, the fraction on the right.
    let padding = (4 - whole.len() % 4) % 4;
    let whole_groups = (whole.len() + padding) / 4;
    let digits: Vec<u8> = std::iter::repeat_n(b'0', padding)
        .chain(whole.bytes())
        .chain(fraction.bytes())
        .chain(std::iter::repeat_n(b'0', (4 - fraction.len() % 4) % 4))
        .map(|b| b - b'0')
        .collect();
    let mut groups: Vec<i16> = digits
        .chunks(4)
        .map(|g| g.iter().fold(0i16, |n, d| n * 10 + i16::from(*d)))
        .collect();
    let mut weight = i32::try_from(whole_groups).map_err(|_| invalid())? - 1;
    let leading = groups.iter().take_while(|g| **g == 0).count();
    groups.drain(..leading);
    weight -= i32::try_from(leading).map_err(|_| invalid())?;
    while groups.last() == Some(&0) {
        groups.pop();
    }
    if groups.is_empty() {
        weight = 0;
    }
    out.put_i16(i16::try_from(groups.len()).map_err(|_| invalid())?);
    out.put_i16(i16::try_from(weight).map_err(|_| invalid())?);
    out.put_u16(if negative && !groups.is_empty() {
        NUMERIC_NEGATIVE
    } else {
        0
    });
    out.put_u16(scale);
    for group in groups {
        out.put_i16(group);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn numeric(decimal: &str) -> Vec<u8> {
        let mut out = BytesMut::new();
        encode_numeric(decimal, &mut out).unwrap();
        out.to_vec()
    }

    /// The bytes PostgreSQL's `numeric_send` gives for the same value.
    fn sent(ndigits: i16, weight: i16, sign: u16, scale: u16, digits: &[i16]) -> Vec<u8> {
        let mut out = BytesMut::new();
        out.put_i16(ndigits);
        out.put_i16(weight);
        out.put_u16(sign);
        out.put_u16(scale);
        for digit in digits {
            out.put_i16(*digit);
        }
        out.to_vec()
    }

    #[test]
    fn numerics_are_written_as_postgres_sends_them() {
        assert_eq!(numeric("0"), sent(0, 0, 0, 0, &[]));
        assert_eq!(numeric("0.00"), sent(0, 0, 0, 2, &[]));
        assert_eq!(numeric("-0.0"), sent(0, 0, 0, 1, &[]));
        assert_eq!(numeric("12345.678"), sent(3, 1, 0, 3, &[1, 2345, 6780]));
        assert_eq!(numeric("-1.5"), sent(2, 0, NUMERIC_NEGATIVE, 1, &[1, 5000]));
        assert_eq!(numeric("0.0001"), sent(1, -1, 0, 4, &[1]));
        assert_eq!(numeric("100000000"), sent(1, 2, 0, 0, &[1]));
        assert_eq!(numeric(".5"), sent(1, -1, 0, 1, &[5000]));
    }

    #[test]
    fn what_is_not_a_decimal_is_refused() {
        for text in ["", "-", ".", "1e5", "1.2.3", "NaN", "+1"] {
            assert!(
                encode_numeric(text, &mut BytesMut::new()).is_err(),
                "{text}"
            );
        }
    }

    #[test]
    fn nanoseconds_round_half_to_even() {
        assert_eq!(nanos_to_micros(1_499), 1);
        assert_eq!(nanos_to_micros(1_500), 2);
        assert_eq!(nanos_to_micros(2_500), 2);
        assert_eq!(nanos_to_micros(2_501), 3);
        assert_eq!(nanos_to_micros(-1_500), -2);
        assert_eq!(nanos_to_micros(-2_500), -2);
    }
}
