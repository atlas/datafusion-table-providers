use crate::{arrow_record_batch_gen::*, docker::RunningContainer};
use arrow::{
    array::{
        Array, Decimal128Array, Decimal128Builder, Int32Array, ListArray, ListBuilder, RecordBatch,
        StringArray, StructArray,
    },
    datatypes::{DataType, Field, Schema, SchemaRef},
};
use datafusion::logical_expr::CreateExternalTable;
use datafusion::physical_plan::collect;
use datafusion::{catalog::TableProvider, execution::context::SessionContext};
use datafusion::{catalog::TableProviderFactory, logical_expr::dml::InsertOp};
use datafusion::{
    common::{Constraints, ToDFSchema},
    datasource::memory::MemorySourceConfig,
};

use datafusion_federation::schema_cast::record_convert::try_cast_to;

use datafusion_table_providers::{
    postgres::{DynPostgresConnectionPool, PostgresTableProviderFactory},
    sql::db_connection_pool::postgrespool::PostgresConnectionPool,
    sql::sql_provider_datafusion::{get_stream, SqlTable},
    util::secrets::to_secret_map,
    UnsupportedTypeAction,
};
use rstest::{fixture, rstest};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, MutexGuard};

mod common;
mod copy;
mod schema;
#[cfg(any(feature = "postgres-native-tls", feature = "postgres-rustls"))]
mod tls;

async fn arrow_postgres_round_trip(
    port: usize,
    arrow_record: RecordBatch,
    source_schema: SchemaRef,
    table_name: &str,
) {
    let factory = PostgresTableProviderFactory::new();
    let ctx = SessionContext::new();
    let cmd = CreateExternalTable {
        schema: Arc::new(arrow_record.schema().to_dfschema().expect("to df schema")),
        name: table_name.into(),
        locations: vec![],
        file_type: "".to_string(),
        table_partition_cols: vec![],
        if_not_exists: false,
        definition: None,
        order_exprs: vec![],
        unbounded: false,
        options: common::get_pg_params(port),
        constraints: Constraints::default(),
        column_defaults: HashMap::new(),
        temporary: false,
        or_replace: false,
    };
    let table_provider = factory
        .create(&ctx.state(), &cmd)
        .await
        .expect("table provider created");

    let ctx = SessionContext::new();
    let mem_exec = MemorySourceConfig::try_new_exec(
        &[vec![arrow_record.clone()]],
        arrow_record.schema(),
        None,
    )
    .expect("memory exec created");
    let insert_plan = table_provider
        .insert_into(&ctx.state(), mem_exec, InsertOp::Append)
        .await
        .expect("insert plan created");

    let _ = collect(insert_plan, ctx.task_ctx())
        .await
        .expect("insert done");
    ctx.register_table(table_name, table_provider)
        .expect("Table should be registered");
    let sql = format!("SELECT * FROM {table_name}");
    let df = ctx
        .sql(&sql)
        .await
        .expect("DataFrame should be created from query");

    let record_batch = df.collect().await.expect("RecordBatch should be collected");

    tracing::debug!("Original Arrow Record Batch: {:?}", arrow_record.columns());
    tracing::debug!(
        "Postgres returned Record Batch: {:?}",
        record_batch[0].columns()
    );

    let casted_result =
        try_cast_to(record_batch[0].clone(), source_schema).expect("Failed to cast record batch");

    // Check results
    assert_eq!(record_batch.len(), 1);
    assert_eq!(record_batch[0].num_rows(), arrow_record.num_rows());
    assert_eq!(record_batch[0].num_columns(), arrow_record.num_columns());

    assert_eq!(arrow_record, casted_result);
}

struct ContainerManager {
    port: usize,
    claimed: bool,
    running_container: Option<RunningContainer>,
}

impl Drop for ContainerManager {
    fn drop(&mut self) {
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(stop_container(self.running_container.take(), self.port));
    }
}

async fn stop_container(running_container: Option<RunningContainer>, port: usize) {
    println!("Stopping Postgres container on port {}", port);
    if let Some(running_container) = running_container {
        if let Err(e) = running_container.stop().await {
            tracing::error!("Error stopping container: {}", e);
        };
    }
}

#[fixture]
#[once]
fn container_manager() -> Mutex<ContainerManager> {
    Mutex::new(ContainerManager {
        port: crate::get_random_port(),
        claimed: false,
        running_container: None,
    })
}

async fn start_container(manager: &mut MutexGuard<'_, ContainerManager>) {
    let running_container = common::start_postgres_docker_container(manager.port)
        .await
        .expect("Postgres container to start");

    manager.running_container = Some(running_container);

    tracing::debug!("Container started");
}

#[rstest]
#[case::binary(get_arrow_binary_record_batch(), "binary")]
#[case::int(get_arrow_int_record_batch(), "int")]
#[case::float(get_arrow_float_record_batch(), "float")]
#[case::utf8(get_arrow_utf8_record_batch(), "utf8")]
#[case::time(get_arrow_time_record_batch(), "time")]
#[case::timestamp(get_arrow_timestamp_record_batch(), "timestamp")]
#[case::date(get_arrow_date_record_batch(), "date")]
#[case::struct_type(get_arrow_struct_record_batch(), "struct")]
#[case::decimal(get_arrow_decimal_record_batch(), "decimal")]
#[case::interval(get_arrow_interval_record_batch(), "interval")]
#[case::duration(get_arrow_duration_record_batch(), "duration")]
#[case::list(get_arrow_list_record_batch(), "list")]
#[case::null(get_arrow_null_record_batch(), "null")]
#[case::bytea_array(get_arrow_bytea_array_record_batch(), "bytea_array")]
#[test_log::test(tokio::test)]
async fn test_arrow_postgres_roundtrip(
    container_manager: &Mutex<ContainerManager>,
    #[case] arrow_result: (RecordBatch, SchemaRef),
    #[case] table_name: &str,
) {
    let mut container_manager = container_manager.lock().await;
    if !container_manager.claimed {
        container_manager.claimed = true;
        start_container(&mut container_manager).await;
    }

    arrow_postgres_round_trip(
        container_manager.port,
        arrow_result.0,
        arrow_result.1,
        &format!("{table_name}_types"),
    )
    .await;
}

#[rstest]
#[test_log::test(tokio::test)]
async fn test_postgres_copy_writes(container_manager: &Mutex<ContainerManager>) {
    let mut container_manager = container_manager.lock().await;
    if !container_manager.claimed {
        container_manager.claimed = true;
        start_container(&mut container_manager).await;
    }

    copy::test_postgres_copy_writes_composites_arrays_and_jsonb(container_manager.port).await;
    copy::test_postgres_copy_resolves_conflicts(container_manager.port).await;
    copy::test_postgres_copy_falls_back_to_insert(container_manager.port).await;
}

#[rstest]
#[test_log::test(tokio::test)]
async fn test_arrow_postgres_one_way(container_manager: &Mutex<ContainerManager>) {
    let mut container_manager = container_manager.lock().await;
    if !container_manager.claimed {
        container_manager.claimed = true;
        start_container(&mut container_manager).await;
    }

    test_postgres_enum_type(container_manager.port).await;
    test_postgres_numeric_type(container_manager.port).await;
    test_postgres_numeric_array_type(container_manager.port).await;
    test_postgres_jsonb_type(container_manager.port).await;
    test_postgres_json_type(container_manager.port).await;
    test_postgres_jsonb_list_struct_with_projected_schema(container_manager.port).await;
    test_postgres_json_list_struct_with_projected_schema(container_manager.port).await;
    test_postgres_composite_array_list_struct(container_manager.port).await;
    test_postgres_domain_types(container_manager.port).await;
    test_postgres_nested_composites(container_manager.port).await;
    test_postgres_sort_limit(container_manager.port).await;
    test_postgres_unconstrained_numeric_precision(container_manager.port).await;
    test_postgres_timestamps_in_microseconds(container_manager.port).await;
}

/// A domain is read as the type it is over, wherever it appears: a column, a domain of a
/// domain, a composite's attribute, an array's element, and a domain over a composite held
/// in an array — the shape a list of structs takes when each element carries a check.
async fn test_postgres_domain_types(port: usize) {
    let pool = common::get_postgres_connection_pool(port)
        .await
        .expect("Postgres connection pool should be created");
    let db_conn = pool
        .connect_direct()
        .await
        .expect("Connection should be established");

    db_conn
        .conn
        .batch_execute(
            "DROP SCHEMA IF EXISTS domains CASCADE;
            CREATE SCHEMA domains;
            CREATE DOMAIN domains.positive AS integer CHECK (VALUE > 0);
            CREATE DOMAIN domains.small AS domains.positive CHECK (VALUE < 100);
            CREATE DOMAIN domains.code AS varchar(8);
            CREATE DOMAIN domains.price AS numeric(10,2);
            CREATE TYPE domains.line_item AS (sku domains.code, qty domains.positive);
            CREATE DOMAIN domains.checked_item AS domains.line_item
                CHECK (VALUE IS NULL OR (VALUE).qty IS NOT NULL);
            CREATE TABLE domains.orders (
                id domains.small PRIMARY KEY,
                code domains.code,
                price domains.price,
                item domains.checked_item,
                items domains.checked_item[],
                counts domains.positive[]
            );
            INSERT INTO domains.orders VALUES
                (1, 'A1', 9.99, ROW('a', 2),
                 ARRAY[ROW('a', 2), ROW('b', 1)]::domains.checked_item[], ARRAY[1, 2]),
                (2, NULL, NULL, NULL, ARRAY[]::domains.checked_item[], NULL);",
        )
        .await
        .expect("Domain fixtures should be created");

    let sqltable_pool: Arc<DynPostgresConnectionPool> = Arc::new(pool);
    let table = SqlTable::new("postgres", &sqltable_pool, "domains.orders")
        .await
        .expect("SqlTable should infer a schema through domains");

    let item = DataType::Struct(
        vec![
            Field::new("sku", DataType::Utf8, true),
            Field::new("qty", DataType::Int32, true),
        ]
        .into(),
    );
    let inferred: Vec<(String, DataType)> = table
        .schema()
        .fields()
        .iter()
        .map(|f| (f.name().clone(), f.data_type().clone()))
        .collect();
    let expected = vec![
        ("id".to_string(), DataType::Int32),
        ("code".to_string(), DataType::Utf8),
        ("price".to_string(), DataType::Decimal128(10, 2)),
        ("item".to_string(), item.clone()),
        (
            "items".to_string(),
            DataType::List(Arc::new(Field::new("item", item, true))),
        ),
        (
            "counts".to_string(),
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
        ),
    ];
    assert_eq!(inferred, expected, "each domain infers as its base type");

    let source_type = |name: &str| {
        table
            .schema()
            .field_with_name(name)
            .unwrap()
            .metadata()
            .get(datafusion_table_providers::SOURCE_TYPE_METADATA_KEY)
            .cloned()
    };
    assert_eq!(
        source_type("price").as_deref(),
        Some("domains.price"),
        "the source type still names the domain"
    );

    let base_type = |name: &str| {
        table
            .schema()
            .field_with_name(name)
            .unwrap()
            .metadata()
            .get(datafusion_table_providers::SOURCE_BASE_TYPE_METADATA_KEY)
            .cloned()
    };
    assert_eq!(base_type("price").as_deref(), Some("numeric(10,2)"));
    assert_eq!(
        base_type("id").as_deref(),
        Some("integer"),
        "a domain of a domain is over the innermost base"
    );
    assert_eq!(base_type("code").as_deref(), Some("character varying(8)"));
    assert_eq!(
        base_type("items"),
        None,
        "an array of domains is an array, not a domain"
    );

    let ctx = SessionContext::new();
    ctx.register_table("orders", Arc::new(table))
        .expect("Table should be registered");
    let batches = ctx
        .sql("SELECT * FROM orders ORDER BY id")
        .await
        .expect("DataFrame should be created from query")
        .collect()
        .await
        .expect("Rows of domain columns should decode");

    let printed = datafusion::arrow::util::pretty::pretty_format_batches(&batches)
        .unwrap()
        .to_string();
    let expected = "\
+----+------+-------+----------------+------------------------------------------+--------+
| id | code | price | item           | items                                    | counts |
+----+------+-------+----------------+------------------------------------------+--------+
| 1  | A1   | 9.99  | {sku: a, qty: 2} | [{sku: a, qty: 2}, {sku: b, qty: 1}]   | [1, 2] |
| 2  |      |       |                |  []                                      |        |
+----+------+-------+----------------+------------------------------------------+--------+";
    // Compared cell by cell rather than as a table, so column widths don't matter.
    let cells = |table: &str| -> Vec<Vec<String>> {
        table
            .lines()
            .filter(|line| line.starts_with('|'))
            .map(|line| line.split('|').map(|c| c.trim().to_string()).collect())
            .collect()
    };
    assert_eq!(cells(&printed), cells(expected), "{printed}");

    db_conn
        .conn
        .batch_execute("DROP SCHEMA domains CASCADE;")
        .await
        .expect("Domain fixtures should be dropped");
}

/// Composites nest to any depth, and arrays of anything but the common scalars decode
/// too: a struct holding a struct and a list of structs, each through a domain, with members
/// of most scalar types, and nulls at every level. The layout is the one a table of
/// checked, nested records takes.
async fn test_postgres_nested_composites(port: usize) {
    let pool = common::get_postgres_connection_pool(port)
        .await
        .expect("Postgres connection pool should be created");
    let db_conn = pool
        .connect_direct()
        .await
        .expect("Connection should be established");

    db_conn
        .conn
        .batch_execute(
            "DROP SCHEMA IF EXISTS nested CASCADE;
            CREATE SCHEMA nested;
            CREATE TYPE nested.owner AS (name text, since timestamptz);
            CREATE DOMAIN nested.checked_owner AS nested.owner
                CHECK (VALUE IS NOT DISTINCT FROM NULL OR (VALUE).name IS NOT NULL);
            CREATE TYPE nested.tag AS (code text, n bigint);
            CREATE DOMAIN nested.checked_tag AS nested.tag;
            CREATE TYPE nested.record AS (
                id uuid, flag boolean, count bigint, ratio double precision, day date,
                doc jsonb, amount numeric(10,2),
                owner nested.checked_owner, tags nested.checked_tag[]
            );
            CREATE DOMAIN nested.element AS nested.record CHECK (VALUE IS DISTINCT FROM NULL);
            CREATE TABLE nested.records (
                id int PRIMARY KEY,
                one nested.record,
                many nested.element[],
                texts varchar(10)[],
                times timestamptz[]
            );
            INSERT INTO nested.records VALUES
                (1,
                 ROW('00000000-0000-0000-0000-000000000001', true, 7, 0.5, '2026-09-28',
                     '{\"a\": 1}', 12.34,
                     ROW('ann', '2026-09-28 12:00:00+00')::nested.owner,
                     ARRAY[ROW('x', 1)::nested.tag, ROW('y', NULL)::nested.tag])::nested.record,
                 ARRAY[
                     ROW('00000000-0000-0000-0000-000000000002', false, NULL, NULL, NULL, NULL,
                         NULL, NULL, ARRAY[]::nested.checked_tag[])::nested.record
                 ]::nested.element[],
                 ARRAY['a', NULL]::varchar(10)[],
                 ARRAY['2026-09-28 12:00:00+00'::timestamptz]),
                (2, NULL, ARRAY[]::nested.element[], NULL, NULL),
                (3, ROW(NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)::nested.record,
                 NULL, ARRAY[]::varchar(10)[], NULL);",
        )
        .await
        .expect("Nested fixtures should be created");

    let sqltable_pool: Arc<DynPostgresConnectionPool> = Arc::new(pool);
    let table = SqlTable::new("postgres", &sqltable_pool, "nested.records")
        .await
        .expect("SqlTable should infer a schema through nested composites");

    let list = |item: DataType| DataType::List(Arc::new(Field::new("item", item, true)));
    let utc = DataType::Timestamp(
        datafusion::arrow::datatypes::TimeUnit::Microsecond,
        Some("UTC".into()),
    );
    let record = DataType::Struct(
        vec![
            Field::new("id", DataType::Utf8, true),
            Field::new("flag", DataType::Boolean, true),
            Field::new("count", DataType::Int64, true),
            Field::new("ratio", DataType::Float64, true),
            Field::new("day", DataType::Date32, true),
            Field::new("doc", DataType::Utf8, true),
            Field::new("amount", DataType::Decimal128(10, 2), true),
            Field::new(
                "owner",
                DataType::Struct(
                    vec![
                        Field::new("name", DataType::Utf8, true),
                        Field::new("since", utc.clone(), true),
                    ]
                    .into(),
                ),
                true,
            ),
            Field::new(
                "tags",
                list(DataType::Struct(
                    vec![
                        Field::new("code", DataType::Utf8, true),
                        Field::new("n", DataType::Int64, true),
                    ]
                    .into(),
                )),
                true,
            ),
        ]
        .into(),
    );
    let types = |schema: SchemaRef| -> Vec<(String, DataType)> {
        schema
            .fields()
            .iter()
            .map(|f| (f.name().clone(), f.data_type().clone()))
            .collect()
    };
    let expected = vec![
        ("id".to_string(), DataType::Int32),
        ("one".to_string(), record.clone()),
        ("many".to_string(), list(record)),
        ("texts".to_string(), list(DataType::Utf8)),
        ("times".to_string(), list(utc)),
    ];
    assert_eq!(
        types(table.schema()),
        expected,
        "nested types infer in full"
    );

    let ctx = SessionContext::new();
    ctx.register_table("records", Arc::new(table))
        .expect("Table should be registered");
    let batches = ctx
        .sql("SELECT * FROM records ORDER BY id")
        .await
        .expect("DataFrame should be created from query")
        .collect()
        .await
        .expect("Rows of nested composites should decode");
    assert_eq!(
        types(batches[0].schema()),
        expected,
        "rows decode into exactly the inferred types"
    );

    let printed = datafusion::arrow::util::pretty::pretty_format_batches(&batches)
        .unwrap()
        .to_string();
    let cells: Vec<Vec<String>> = printed
        .lines()
        .filter(|line| line.starts_with('|'))
        .skip(1)
        .map(|line| {
            line.trim_matches('|')
                .split(" | ")
                .map(|c| c.trim().to_string())
                .collect()
        })
        .collect();
    println!("{printed}");
    assert_eq!(
        cells,
        vec![
            vec![
                "1".to_string(),
                "{id: 00000000-0000-0000-0000-000000000001, flag: true, count: 7, ratio: 0.5, \
                 day: 2026-09-28, doc: {\"a\": 1}, amount: 12.34, \
                 owner: {name: ann, since: 2026-09-28T12:00:00Z}, \
                 tags: [{code: x, n: 1}, {code: y, n: }]}"
                    .to_string(),
                "[{id: 00000000-0000-0000-0000-000000000002, flag: false, count: , ratio: , \
                 day: , doc: , amount: , owner: , tags: []}]"
                    .to_string(),
                "[a, ]".to_string(),
                "[2026-09-28T12:00:00Z]".to_string(),
            ],
            vec![
                "2".to_string(),
                String::new(),
                "[]".to_string(),
                String::new(),
                String::new(),
            ],
            vec![
                "3".to_string(),
                "{id: , flag: , count: , ratio: , day: , doc: , amount: , owner: , tags: }"
                    .to_string(),
                String::new(),
                "[]".to_string(),
                String::new(),
            ],
        ],
    );

    // A member of a type nothing decodes fails the table, and does not panic.
    db_conn
        .conn
        .batch_execute(
            "CREATE TYPE nested.odd AS (at point);
            CREATE TABLE nested.odds (id int, odd nested.odd);",
        )
        .await
        .expect("Unsupported fixture should be created");
    let Err(err) = SqlTable::new("postgres", &sqltable_pool, "nested.odds").await else {
        panic!("A composite with an unsupported member should be refused");
    };
    assert!(err.to_string().contains("odd"), "{err}");

    db_conn
        .conn
        .batch_execute("DROP SCHEMA nested CASCADE;")
        .await
        .expect("Nested fixtures should be dropped");
}

/// Timestamps read as microseconds, PostgreSQL's own precision, so every value it holds
/// keeps its instant: before 1970, before 1677 and after 2262, where nanoseconds since
/// 1970 overflow, at the top level and in a composite alike.
async fn test_postgres_timestamps_in_microseconds(port: usize) {
    let pool = common::get_postgres_connection_pool(port)
        .await
        .expect("Postgres connection pool should be created");
    let db_conn = pool
        .connect_direct()
        .await
        .expect("Connection should be established");
    db_conn
        .conn
        .batch_execute(
            "DROP SCHEMA IF EXISTS times CASCADE;
            CREATE SCHEMA times;
            CREATE TYPE times.stamped AS (at timestamptz);
            CREATE TABLE times.values (id int, local timestamp, utc timestamptz, inner_at times.stamped);
            INSERT INTO times.values VALUES
                (1, '1969-12-31 23:59:59.5', '1969-12-31 23:59:59.5+00', ROW('1969-12-31 23:59:59.5+00')),
                (2, '1500-01-01 00:00:00', '1500-01-01 00:00:00+00', ROW('1500-01-01 00:00:00+00')),
                (3, '3000-01-01 00:00:00.000001', '3000-01-01 00:00:00.000001+00', ROW('3000-01-01 00:00:00.000001+00')),
                (4, NULL, NULL, ROW(NULL));",
        )
        .await
        .expect("Timestamp fixtures should be created");

    let sqltable_pool: Arc<DynPostgresConnectionPool> = Arc::new(pool);
    let table = SqlTable::new("postgres", &sqltable_pool, "times.values")
        .await
        .expect("SqlTable should infer the timestamps' schema");
    let micros = |zone: Option<&str>| {
        DataType::Timestamp(
            datafusion::arrow::datatypes::TimeUnit::Microsecond,
            zone.map(Into::into),
        )
    };
    let ctx = SessionContext::new();
    ctx.register_table("stamps", Arc::new(table))
        .expect("Table should be registered");
    let batches = ctx
        .sql("SELECT * FROM stamps ORDER BY id")
        .await
        .expect("DataFrame should be created from query")
        .collect()
        .await
        .expect("Timestamps should decode");
    let schema = batches[0].schema();
    assert_eq!(schema.field(1).data_type(), &micros(None));
    assert_eq!(schema.field(2).data_type(), &micros(Some("UTC")));
    assert_eq!(
        schema.field(3).data_type(),
        &DataType::Struct(vec![Field::new("at", micros(Some("UTC")), true)].into())
    );
    let printed = datafusion::arrow::util::pretty::pretty_format_batches(&batches)
        .unwrap()
        .to_string();
    let rows: Vec<String> = printed
        .lines()
        .filter(|line| line.starts_with('|'))
        .skip(1)
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect();
    assert_eq!(
        rows,
        vec![
            "| 1 | 1969-12-31T23:59:59.500 | 1969-12-31T23:59:59.500Z | {at: 1969-12-31T23:59:59.500Z} |",
            "| 2 | 1500-01-01T00:00:00 | 1500-01-01T00:00:00Z | {at: 1500-01-01T00:00:00Z} |",
            "| 3 | 3000-01-01T00:00:00.000001 | 3000-01-01T00:00:00.000001Z | {at: 3000-01-01T00:00:00.000001Z} |",
            "| 4 | | | {at: } |",
        ],
        "{printed}"
    );

    db_conn
        .conn
        .batch_execute("DROP SCHEMA times CASCADE;")
        .await
        .expect("Timestamp fixtures should be dropped");
}

/// An unconstrained `numeric` column (what `max()`, `avg()` and arithmetic over `numeric`
/// return) has no fixed scale, so the Arrow column scale must not be taken from the first
/// row: every later row carrying more decimals was rounded down to it.
///
/// The fixture order matters — a whole number first, longer values after. Ordered the other
/// way round the values happen to survive even without the fix.
async fn test_postgres_unconstrained_numeric_precision(port: usize) {
    let pool = common::get_postgres_connection_pool(port)
        .await
        .expect("Postgres connection pool should be created");
    let sqltable_pool: Arc<DynPostgresConnectionPool> = Arc::new(pool);
    let conn = sqltable_pool.connect().await.expect("connect should work");
    let async_conn = conn.as_async().expect("should be async connection");

    // No projected schema: the scale can only come from the column itself.
    let stream = async_conn
        .query_arrow(
            "SELECT v FROM (VALUES (20::numeric),(17.685::numeric),(15.334::numeric)) t(v)",
            &[],
            None,
        )
        .await
        .expect("query should work");
    let batches: Vec<RecordBatch> = futures::StreamExt::collect::<Vec<_>>(stream)
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("batches should be readable");

    let scale: u32 = 20;
    let expected: Vec<i128> = vec![
        20 * 10i128.pow(scale),
        17_685 * 10i128.pow(scale - 3),
        15_334 * 10i128.pow(scale - 3),
    ];

    let got: Vec<i128> = batches
        .iter()
        .flat_map(|batch| {
            let column = batch
                .column(0)
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .expect("v should be Decimal128");
            assert_eq!(
                column.scale(),
                20,
                "unconstrained numeric must use the fixed default scale, not the first row's"
            );
            (0..column.len())
                .map(|i| column.value(i))
                .collect::<Vec<_>>()
        })
        .collect();

    assert_eq!(
        got, expected,
        "every numeric value must survive exactly, independent of row order"
    );
}

async fn test_postgres_sort_limit(port: usize) {
    let ctx = SessionContext::new();
    let pool = common::get_postgres_connection_pool(port)
        .await
        .expect("Postgres connection pool should be created");

    let db_conn = pool
        .connect_direct()
        .await
        .expect("Connection should be established");

    // Prepare table: 20 rows with id = 1..=20.
    let _ = db_conn
        .conn
        .execute("DROP TABLE IF EXISTS sort_limit_test", &[])
        .await
        .expect("table should be droppable");
    let _ = db_conn
        .conn
        .execute(
            "CREATE TABLE sort_limit_test (id INT NOT NULL, label TEXT NOT NULL)",
            &[],
        )
        .await
        .expect("CREATE TABLE should succeed");
    let values: Vec<String> = (1..=20).map(|i| format!("({i}, 'row-{i:02}')")).collect();
    let insert_stmt = format!(
        "INSERT INTO sort_limit_test (id, label) VALUES {}",
        values.join(",")
    );
    let _ = db_conn
        .conn
        .execute(&insert_stmt, &[])
        .await
        .expect("INSERT should succeed");

    let sqltable_pool: Arc<DynPostgresConnectionPool> = Arc::new(pool);
    let table = SqlTable::new("postgres", &sqltable_pool, "sort_limit_test")
        .await
        .expect("Table should be created");
    ctx.register_table("sort_limit_test", Arc::new(table))
        .expect("Table should be registered");

    // 1. ORDER BY DESC + LIMIT 5 must return exactly 5 rows, top-down.
    let df = ctx
        .sql("SELECT id FROM sort_limit_test ORDER BY id DESC LIMIT 5")
        .await
        .expect("SQL should parse");
    let batches = df.collect().await.expect("query should succeed");
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 5, "LIMIT 5 must return exactly 5 rows");
    let col = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("id column is Int32");
    let got: Vec<i32> = (0..col.len()).map(|i| col.value(i)).collect();
    assert_eq!(got, vec![20, 19, 18, 17, 16]);

    // 2. ORDER BY + LIMIT with WHERE.
    let df = ctx
        .sql("SELECT id FROM sort_limit_test WHERE id > 10 ORDER BY id ASC LIMIT 3")
        .await
        .expect("SQL should parse");
    let batches = df.collect().await.expect("query should succeed");
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 3);
    let col = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let got: Vec<i32> = (0..col.len()).map(|i| col.value(i)).collect();
    assert_eq!(got, vec![11, 12, 13]);

    // 3. Bare LIMIT (no ORDER BY) must still cap rows.
    let df = ctx
        .sql("SELECT id FROM sort_limit_test LIMIT 7")
        .await
        .expect("SQL should parse");
    let batches = df.collect().await.expect("query should succeed");
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 7);
}

async fn test_postgres_enum_type(port: usize) {
    let extra_stmt = Some("CREATE TYPE mood AS ENUM ('happy', 'sad', 'neutral');");
    let create_table_stmt = "
    CREATE TABLE person_mood (
    mood_status mood NOT NULL
    );";

    let insert_table_stmt = "
    INSERT INTO person_mood (mood_status) VALUES ('happy'), ('sad'), ('neutral');
    ";

    let (expected_record, _) = get_arrow_dictionary_array_record_batch();

    arrow_postgres_one_way(
        port,
        "person_mood",
        create_table_stmt,
        insert_table_stmt,
        extra_stmt,
        expected_record,
        UnsupportedTypeAction::default(),
    )
    .await;
}

async fn test_postgres_numeric_type(port: usize) {
    let extra_stmt = None;
    let create_table_stmt = "
    CREATE TABLE numeric_values (
    first_column NUMERIC,  -- No precision specified
    second_column NUMERIC  -- No precision specified
);";

    let insert_table_stmt = "
    INSERT INTO numeric_values (first_column, second_column) VALUES
(1.0917217805754313, 0.00000000000000000000),
(0.97824560830666753739, 1220.9175000000000000),
(1.0917217805754313, 52.9533333333333333);
    ";

    let schema = Arc::new(Schema::new(vec![
        Field::new("first_column", DataType::Decimal128(38, 20), true),
        Field::new("second_column", DataType::Decimal128(38, 20), true),
    ]));

    let expected_record = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(
                Decimal128Array::from(vec![
                    109172178057543130000i128,
                    97824560830666753739i128,
                    109172178057543130000i128,
                ])
                .with_precision_and_scale(38, 20)
                .unwrap(),
            ),
            Arc::new(
                Decimal128Array::from(vec![
                    0i128,
                    122091750000000000000000i128,
                    5295333333333333330000i128,
                ])
                .with_precision_and_scale(38, 20)
                .unwrap(),
            ),
        ],
    )
    .expect("Failed to created arrow record batch");

    arrow_postgres_one_way(
        port,
        "numeric_values",
        create_table_stmt,
        insert_table_stmt,
        extra_stmt,
        expected_record,
        UnsupportedTypeAction::default(),
    )
    .await;
}

async fn test_postgres_numeric_array_type(port: usize) {
    let create_table_stmt = "
    CREATE TABLE numeric_array_values (
    numeric_values NUMERIC[]
);";

    let insert_table_stmt = "
    INSERT INTO numeric_array_values (numeric_values) VALUES
(ARRAY[1.2300::NUMERIC, 42::NUMERIC, NULL::NUMERIC, -0.0045::NUMERIC]),
(NULL),
(ARRAY[]::NUMERIC[]),
(ARRAY[100.1::NUMERIC]);
    ";

    let decimal_item_type = DataType::Decimal128(38, 20);
    let schema = Arc::new(Schema::new(vec![Field::new(
        "numeric_values",
        DataType::List(Arc::new(Field::new(
            "item",
            decimal_item_type.clone(),
            true,
        ))),
        true,
    )]));

    let mut numeric_array_builder = ListBuilder::new(
        Decimal128Builder::new()
            .with_precision_and_scale(38, 20)
            .expect("Failed to create Decimal128Builder with expected precision and scale"),
    );

    numeric_array_builder
        .values()
        .append_value(123000000000000000000);
    numeric_array_builder
        .values()
        .append_value(4200000000000000000000);
    numeric_array_builder.values().append_null();
    numeric_array_builder
        .values()
        .append_value(-450000000000000000);
    numeric_array_builder.append(true);
    numeric_array_builder.append_null();
    numeric_array_builder.append(true);
    numeric_array_builder
        .values()
        .append_value(10010000000000000000000);
    numeric_array_builder.append(true);

    let expected_record = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(numeric_array_builder.finish())],
    )
    .expect("Failed to create expected record batch for NUMERIC[]");

    arrow_postgres_one_way(
        port,
        "numeric_array_values",
        create_table_stmt,
        insert_table_stmt,
        None,
        expected_record,
        UnsupportedTypeAction::default(),
    )
    .await;
}

async fn test_postgres_jsonb_type(port: usize) {
    let create_table_stmt = "
    CREATE TABLE jsonb_values (
        id INT PRIMARY KEY,
        data JSONB
    );";

    let insert_table_stmt = r#"
    INSERT INTO jsonb_values (id, data) VALUES
    (1, '{"name": "John", "age": 30}'),
    (2, '{"name": "Jane", "age": 25}'),
    (3, '[1, 2, 3]'),
    (4, 'null'),
    (5, '{"nested": {"key": "value"}}');
    "#;

    let expected_values: Vec<Value> = vec![
        serde_json::from_str(r#"{"name":"John","age":30}"#).unwrap(),
        serde_json::from_str(r#"{"name":"Jane","age":25}"#).unwrap(),
        serde_json::from_str("[1,2,3]").unwrap(),
        serde_json::from_str("null").unwrap(),
        serde_json::from_str(r#"{"nested":{"key":"value"}}"#).unwrap(),
    ];
    let batches = query_postgres_one_way(
        port,
        "jsonb_values",
        create_table_stmt,
        insert_table_stmt,
        None,
        UnsupportedTypeAction::String,
        Some("SELECT data FROM jsonb_values ORDER BY id"),
    )
    .await;
    assert_eq!(batches.len(), 1);

    let col = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("column should be StringArray");

    assert_eq!(col.len(), expected_values.len());
    for (i, expected) in expected_values.iter().enumerate() {
        let actual: Value =
            serde_json::from_str(col.value(i)).expect("actual value should be valid JSON");
        assert_eq!(&actual, expected, "mismatch at row {i}");
    }
}

/// Guards that plain JSON columns (not JSONB) still round-trip as Utf8 through
/// `JsonbRawString` without the serde_json::Value intermediate.
async fn test_postgres_json_type(port: usize) {
    let create_table_stmt = "
    CREATE TABLE json_values (
        id INT PRIMARY KEY,
        data JSON
    );";

    let insert_table_stmt = r#"
    INSERT INTO json_values (id, data) VALUES
    (1, '{"name": "Alice"}'),
    (2, '[1, 2]'),
    (3, 'null');
    "#;

    let expected_values: Vec<Value> = vec![
        serde_json::from_str(r#"{"name":"Alice"}"#).unwrap(),
        serde_json::from_str("[1,2]").unwrap(),
        serde_json::from_str("null").unwrap(),
    ];
    let batches = query_postgres_one_way(
        port,
        "json_values",
        create_table_stmt,
        insert_table_stmt,
        None,
        UnsupportedTypeAction::String,
        Some("SELECT data FROM json_values ORDER BY id"),
    )
    .await;
    assert_eq!(batches.len(), 1);

    let col = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("column should be StringArray");

    assert_eq!(col.len(), expected_values.len());
    for (i, expected) in expected_values.iter().enumerate() {
        let actual: Value =
            serde_json::from_str(col.value(i)).expect("actual value should be valid JSON");
        assert_eq!(&actual, expected, "mismatch at row {i}");
    }
}

async fn test_postgres_json_list_struct_projected(port: usize, sql_type: &str) {
    let table_name = format!("{sql_type}_list_struct_values").to_lowercase();

    let create_table_stmt = format!(
        "CREATE TABLE {table_name} (
            id INT PRIMARY KEY,
            data {sql_type}
        );"
    );

    let insert_table_stmt = format!(
        r#"INSERT INTO {table_name} (id, data) VALUES
            (1, '[{{"id":"u1","email":"one@example.com"}},{{"id":"u2","email":"two@example.com"}}]'),
            (2, '[]'),
            (3, null);
        "#
    );

    let ctx = SessionContext::new();

    let pool = common::get_postgres_connection_pool(port)
        .await
        .expect("Postgres connection pool should be created")
        .with_unsupported_type_action(UnsupportedTypeAction::String);

    let db_conn = pool
        .connect_direct()
        .await
        .expect("Connection should be established");

    let _ = db_conn
        .conn
        .execute(&create_table_stmt, &[])
        .await
        .expect("Postgres table should be created");

    let _ = db_conn
        .conn
        .execute(&insert_table_stmt, &[])
        .await
        .expect("Postgres table data should be inserted");

    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        Field::new(
            "data",
            DataType::List(Arc::new(Field::new(
                "item",
                DataType::Struct(
                    vec![
                        Field::new("id", DataType::Utf8, true),
                        Field::new("email", DataType::Utf8, true),
                    ]
                    .into(),
                ),
                true,
            ))),
            true,
        ),
    ]));

    let sqltable_pool: Arc<DynPostgresConnectionPool> = Arc::new(pool);
    let table = SqlTable::new_with_schema(
        "postgres",
        &sqltable_pool,
        Arc::clone(&projected_schema),
        &table_name,
    );
    ctx.register_table(table_name.as_str(), Arc::new(table))
        .expect("Table should be registered");

    let df = ctx
        .sql(&format!("SELECT id, data FROM {table_name} ORDER BY id"))
        .await
        .expect("DataFrame should be created from query");

    let record_batch = df.collect().await.expect("RecordBatch should be collected");
    assert_eq!(record_batch.len(), 1);
    assert_eq!(record_batch[0].num_rows(), 3);

    let data_col = record_batch[0]
        .column(1)
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("data should decode to ListArray");

    assert!(!data_col.is_null(0));
    assert_eq!(data_col.value_length(0), 2);
    assert!(!data_col.is_null(1));
    assert_eq!(data_col.value_length(1), 0);
    assert!(data_col.is_null(2));

    let row_one_values = data_col.value(0);
    let row_one_struct = row_one_values
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("row 1 values should be StructArray");
    let ids = row_one_struct
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("id field should be StringArray");
    let emails = row_one_struct
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("email field should be StringArray");

    assert_eq!(ids.value(0), "u1");
    assert_eq!(ids.value(1), "u2");
    assert_eq!(emails.value(0), "one@example.com");
    assert_eq!(emails.value(1), "two@example.com");
}

/// Reads a native PostgreSQL array-of-composite column (`composite_type[]`) back as an
/// Arrow `List<Struct>`, including the empty-array and NULL cases.
async fn test_postgres_composite_array_list_struct(port: usize) {
    let table_name = "composite_array_list_struct_values".to_string();

    let create_type_stmt = "
        CREATE TYPE line_item AS (
            sku TEXT,
            qty INT,
            price DOUBLE PRECISION
        );";
    let create_table_stmt = format!(
        "CREATE TABLE {table_name} (
            id INT PRIMARY KEY,
            items line_item[]
        );"
    );
    let insert_table_stmt = format!(
        "INSERT INTO {table_name} (id, items) VALUES
            (1, ARRAY[ROW('a', 2, 9.99), ROW('b', 1, 4.50)]::line_item[]),
            (2, ARRAY[]::line_item[]),
            (3, NULL);"
    );

    let ctx = SessionContext::new();

    let pool = common::get_postgres_connection_pool(port)
        .await
        .expect("Postgres connection pool should be created");

    let db_conn = pool
        .connect_direct()
        .await
        .expect("Connection should be established");

    // The container is shared across the module (`#[fixture] #[once]`), so make setup
    // idempotent: drop any artifacts left by a prior run before recreating them. The
    // table must go first — it depends on the composite type.
    let _ = db_conn
        .conn
        .execute(&format!("DROP TABLE IF EXISTS {table_name};"), &[])
        .await
        .expect("Existing table should be dropped");
    let _ = db_conn
        .conn
        .execute("DROP TYPE IF EXISTS line_item;", &[])
        .await
        .expect("Existing composite type should be dropped");

    let _ = db_conn
        .conn
        .execute(create_type_stmt, &[])
        .await
        .expect("Postgres composite type should be created");
    let _ = db_conn
        .conn
        .execute(&create_table_stmt, &[])
        .await
        .expect("Postgres table should be created");
    let _ = db_conn
        .conn
        .execute(&insert_table_stmt, &[])
        .await
        .expect("Postgres table data should be inserted");

    let item_struct = DataType::Struct(
        vec![
            Field::new("sku", DataType::Utf8, true),
            Field::new("qty", DataType::Int32, true),
            Field::new("price", DataType::Float64, true),
        ]
        .into(),
    );
    let expected_items_type = DataType::List(Arc::new(Field::new("item", item_struct, true)));

    // Register via `SqlTable::new` (no explicit schema) so `get_schema` auto-infers the
    // composite array as List<Struct> from the catalog, exercising the schema SQL +
    // `parse_array_type` composite-element path end to end.
    let sqltable_pool: Arc<DynPostgresConnectionPool> = Arc::new(pool);
    let table = SqlTable::new("postgres", &sqltable_pool, table_name.clone())
        .await
        .expect("SqlTable should infer schema");

    let inferred = table
        .schema()
        .field_with_name("items")
        .expect("items field inferred")
        .data_type()
        .clone();
    assert_eq!(
        inferred, expected_items_type,
        "composite array should auto-infer to List<Struct>"
    );

    ctx.register_table(table_name.as_str(), Arc::new(table))
        .expect("Table should be registered");

    let df = ctx
        .sql(&format!("SELECT id, items FROM {table_name} ORDER BY id"))
        .await
        .expect("DataFrame should be created from query");

    let record_batch = df.collect().await.expect("RecordBatch should be collected");
    assert_eq!(record_batch.len(), 1);
    assert_eq!(record_batch[0].num_rows(), 3);

    let items_col = record_batch[0]
        .column(1)
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("items should decode to ListArray");

    // row 0: two structs, row 1: empty list (not null), row 2: NULL.
    assert!(!items_col.is_null(0));
    assert_eq!(items_col.value_length(0), 2);
    assert!(!items_col.is_null(1));
    assert_eq!(items_col.value_length(1), 0);
    assert!(items_col.is_null(2));

    let row_zero = items_col.value(0);
    let structs = row_zero
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("list values should be StructArray");

    let skus = structs
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("sku field should be StringArray");
    let qtys = structs
        .column(1)
        .as_any()
        .downcast_ref::<arrow::array::Int32Array>()
        .expect("qty field should be Int32Array");
    let prices = structs
        .column(2)
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
        .expect("price field should be Float64Array");

    assert_eq!(skus.value(0), "a");
    assert_eq!(skus.value(1), "b");
    assert_eq!(qtys.value(0), 2);
    assert_eq!(qtys.value(1), 1);
    assert!((prices.value(0) - 9.99).abs() < f64::EPSILON);
    assert!((prices.value(1) - 4.50).abs() < f64::EPSILON);
}

async fn test_postgres_jsonb_list_struct_with_projected_schema(port: usize) {
    test_postgres_json_list_struct_projected(port, "JSONB").await;
}

async fn test_postgres_json_list_struct_with_projected_schema(port: usize) {
    test_postgres_json_list_struct_projected(port, "JSON").await;
}

/// Validates that [`PostgresConnectionPool::new_with_password_provider`] produces
/// a working pool by creating a table, inserting, and querying through the provider path.
#[rstest]
#[test_log::test(tokio::test)]
async fn test_password_provider_pool(container_manager: &Mutex<ContainerManager>) {
    let mut container_manager = container_manager.lock().await;
    if !container_manager.claimed {
        container_manager.claimed = true;
        start_container(&mut container_manager).await;
    }

    let pool = common::get_postgres_pool_with_password_provider(container_manager.port)
        .await
        .expect("Pool with password provider should be created");

    // Verify pool works: get a connection, create a table, insert, query
    let conn = pool
        .connect_direct()
        .await
        .expect("Connection should be established");

    conn.conn
        .execute(
            "CREATE TABLE IF NOT EXISTS password_provider_test (id INT, name TEXT)",
            &[],
        )
        .await
        .expect("Table should be created");

    conn.conn
        .execute(
            "INSERT INTO password_provider_test VALUES (1, 'hello')",
            &[],
        )
        .await
        .expect("Insert should succeed");

    let rows = conn
        .conn
        .query("SELECT id, name FROM password_provider_test", &[])
        .await
        .expect("Query should succeed");

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get::<_, i32>(0), 1);
    assert_eq!(rows[0].get::<_, String>(1), "hello");

    // Also verify it works through the SqlTable (DataFusion) path
    let sqltable_pool: Arc<DynPostgresConnectionPool> = Arc::new(pool);
    let table = SqlTable::new("postgres", &sqltable_pool, "password_provider_test")
        .await
        .expect("SqlTable should be created");

    let ctx = SessionContext::new();
    ctx.register_table("password_provider_test", Arc::new(table))
        .expect("Table should be registered");

    let df = ctx
        .sql("SELECT * FROM password_provider_test")
        .await
        .expect("Query should execute");
    let batches = df.collect().await.expect("Results should be collected");

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);
}

async fn arrow_postgres_one_way(
    port: usize,
    table_name: &str,
    create_table_stmt: &str,
    insert_table_stmt: &str,
    extra_stmt: Option<&str>,
    expected_record: RecordBatch,
    unsupported_type_action: UnsupportedTypeAction,
) {
    let record_batch = query_postgres_one_way(
        port,
        table_name,
        create_table_stmt,
        insert_table_stmt,
        extra_stmt,
        unsupported_type_action,
        None,
    )
    .await;

    assert_eq!(record_batch[0], expected_record);
}

async fn query_postgres_one_way(
    port: usize,
    table_name: &str,
    create_table_stmt: &str,
    insert_table_stmt: &str,
    extra_stmt: Option<&str>,
    unsupported_type_action: UnsupportedTypeAction,
    query: Option<&str>,
) -> Vec<RecordBatch> {
    tracing::debug!("Running tests on {table_name}");
    let ctx = SessionContext::new();

    let pool = common::get_postgres_connection_pool(port)
        .await
        .expect("Postgres connection pool should be created")
        .with_unsupported_type_action(unsupported_type_action);

    let db_conn = pool
        .connect_direct()
        .await
        .expect("Connection should be established");

    if let Some(extra_stmt) = extra_stmt {
        let _ = db_conn
            .conn
            .execute(extra_stmt, &[])
            .await
            .expect("Statement should be created");
    }

    let _ = db_conn
        .conn
        .execute(create_table_stmt, &[])
        .await
        .expect("Postgres table should be created");

    let _ = db_conn
        .conn
        .execute(insert_table_stmt, &[])
        .await
        .expect("Postgres table data should be inserted");

    // Register datafusion table, test row -> arrow conversion
    let sqltable_pool: Arc<DynPostgresConnectionPool> = Arc::new(pool);
    let table = SqlTable::new("postgres", &sqltable_pool, table_name)
        .await
        .expect("Table should be created");
    ctx.register_table(table_name, Arc::new(table))
        .expect("Table should be registered");
    let sql = query
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| format!("SELECT * FROM {table_name}"));
    let df = ctx
        .sql(&sql)
        .await
        .expect("DataFrame should be created from query");

    df.collect().await.expect("RecordBatch should be collected")
}

#[rstest]
#[test_log::test(tokio::test)]
async fn test_postgres_io_runtime_segregation(container_manager: &Mutex<ContainerManager>) {
    let mut container_manager = container_manager.lock().await;
    if !container_manager.claimed {
        container_manager.claimed = true;
        start_container(&mut container_manager).await;
    }

    // Create a separate IO runtime
    let io_runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("IO runtime should be created");

    let pool = common::get_postgres_connection_pool(container_manager.port)
        .await
        .expect("pool created")
        .with_io_runtime(io_runtime.handle().clone());

    // Verify the pool works through the IO runtime
    let sqltable_pool: Arc<DynPostgresConnectionPool> = Arc::new(pool);
    let conn = sqltable_pool.connect().await.expect("connect should work");
    let async_conn = conn.as_async().expect("should be async connection");
    // Execute a simple query to confirm IO runtime is functional
    let stream = async_conn
        .query_arrow("SELECT 1 AS val", &[], None)
        .await
        .expect("query should work");
    let batches: Vec<_> = futures::StreamExt::collect(stream).await;
    assert!(!batches.is_empty(), "should return results via IO runtime");

    io_runtime.shutdown_background();
}

/// A pool given `hostaddr` reaches the server there without resolving `host`, and a query
/// whose rows are dropped unread is cancelled on the server rather than left running.
#[rstest]
#[test_log::test(tokio::test)]
async fn test_postgres_pinned_address_and_cancel_on_drop(
    container_manager: &Mutex<ContainerManager>,
) {
    let mut container_manager = container_manager.lock().await;
    if !container_manager.claimed {
        container_manager.claimed = true;
        start_container(&mut container_manager).await;
    }

    let mut params = common::get_pg_params(container_manager.port);
    // `.invalid` resolves nowhere, so only the pinned address reaches the server.
    params.insert("pg_host".to_string(), "customer.invalid".to_string());
    params.insert("pg_hostaddr".to_string(), "127.0.0.1".to_string());
    let pool = PostgresConnectionPool::new(to_secret_map(params))
        .await
        .expect("a pinned pool should connect without resolving its host");
    let pool: Arc<DynPostgresConnectionPool> = Arc::new(pool);
    let conn = pool.connect().await.expect("connect should work");
    let async_conn = conn.as_async().expect("should be async connection");

    // The first batch arrives at once; the row after the rest takes a minute.
    let mut rows = async_conn
        .query_arrow(
            "SELECT g FROM generate_series(1, 20001) g \
             WHERE g <= 20000 OR (SELECT true FROM pg_sleep(60) WHERE g > 20000)",
            &[],
            None,
        )
        .await
        .expect("query should work");
    futures::StreamExt::next(&mut rows)
        .await
        .expect("a first batch")
        .expect("the first batch should be readable");
    drop(rows);

    let admin = common::get_postgres_connection_pool(container_manager.port)
        .await
        .expect("Postgres connection pool should be created")
        .connect_direct()
        .await
        .expect("Connection should be established");
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let running: i64 = admin
            .conn
            .query_one(
                "SELECT count(*) FROM pg_stat_activity \
                 WHERE state = 'active' AND query LIKE '%pg_sleep(60)%' \
                 AND pid <> pg_backend_pid()",
                &[],
            )
            .await
            .expect("pg_stat_activity should be readable")
            .get(0);
        if running == 0 {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the query whose rows were dropped is still running"
        );
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
}

/// A connection whose stream was dropped before its end is discarded by the pool, so the
/// request to cancel its query never reaches the next statement.
#[rstest]
#[test_log::test(tokio::test)]
async fn test_postgres_cancel_never_reaches_the_next_statement(
    container_manager: &Mutex<ContainerManager>,
) {
    let mut container_manager = container_manager.lock().await;
    if !container_manager.claimed {
        container_manager.claimed = true;
        start_container(&mut container_manager).await;
    }

    let mut params = common::get_pg_params(container_manager.port);
    params.insert("pg_connection_pool_size".to_string(), "1".to_string());
    let pool: Arc<DynPostgresConnectionPool> = Arc::new(
        PostgresConnectionPool::new(to_secret_map(params))
            .await
            .expect("Postgres connection pool should be created"),
    );

    // One row past the first batch: the server has finished the query while the client still
    // holds a row unread, so the cancel for it can only reach whatever runs next. Whether it
    // arrives before or during that statement is a race, so it is run often enough to lose it.
    for _ in 0..30 {
        let mut rows = get_stream(
            Arc::clone(&pool),
            "SELECT g FROM generate_series(1, 4001) g".to_string(),
            Arc::new(Schema::new(vec![Field::new("g", DataType::Int32, true)])),
        )
        .await
        .expect("query should work");
        futures::StreamExt::next(&mut rows)
            .await
            .expect("a first batch")
            .expect("the first batch should be readable");
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        drop(rows);

        let next = get_stream(
            Arc::clone(&pool),
            "SELECT true AS slept FROM pg_sleep(0.3)".to_string(),
            Arc::new(Schema::new(vec![Field::new(
                "slept",
                DataType::Boolean,
                true,
            )])),
        )
        .await
        .expect("the next statement should not be cancelled");
        let batches = futures::StreamExt::collect::<Vec<_>>(next).await;
        assert_eq!(batches.len(), 1);
        assert_eq!(
            batches[0]
                .as_ref()
                .expect("the next statement should not be cancelled")
                .num_rows(),
            1
        );
    }
}

async fn backend_pid(pool: &PostgresConnectionPool) -> i32 {
    pool.connect_direct()
        .await
        .expect("Connection should be established")
        .conn
        .query_one("SELECT pg_backend_pid()", &[])
        .await
        .expect("pg_backend_pid should be readable")
        .get(0)
}

async fn single_connection_pool(port: usize) -> Arc<PostgresConnectionPool> {
    let mut params = common::get_pg_params(port);
    params.insert("pg_connection_pool_size".to_string(), "1".to_string());
    Arc::new(
        PostgresConnectionPool::new(to_secret_map(params))
            .await
            .expect("Postgres connection pool should be created"),
    )
}

/// The pool discards the connection a stream was dropped unfinished on, and keeps the one a
/// stream was read to its end on.
#[rstest]
#[test_log::test(tokio::test)]
async fn test_postgres_a_cancelled_connection_is_discarded(
    container_manager: &Mutex<ContainerManager>,
) {
    let mut container_manager = container_manager.lock().await;
    if !container_manager.claimed {
        container_manager.claimed = true;
        start_container(&mut container_manager).await;
    }

    let pool = single_connection_pool(container_manager.port).await;
    let before = backend_pid(&pool).await;

    let mut rows = get_stream(
        Arc::clone(&pool) as Arc<DynPostgresConnectionPool>,
        "SELECT g FROM generate_series(1, 4001) g".to_string(),
        Arc::new(Schema::new(vec![Field::new("g", DataType::Int32, true)])),
    )
    .await
    .expect("query should work");
    futures::StreamExt::next(&mut rows)
        .await
        .expect("a first batch")
        .expect("the first batch should be readable");
    drop(rows);
    let after_drop = backend_pid(&pool).await;
    assert_ne!(after_drop, before, "a cancelled connection is not reused");

    let read = get_stream(
        Arc::clone(&pool) as Arc<DynPostgresConnectionPool>,
        "SELECT g FROM generate_series(1, 4001) g".to_string(),
        Arc::new(Schema::new(vec![Field::new("g", DataType::Int32, true)])),
    )
    .await
    .expect("query should work");
    let batches = futures::StreamExt::collect::<Vec<_>>(read).await;
    assert!(batches.iter().all(Result::is_ok));
    assert_eq!(backend_pid(&pool).await, after_drop, "a finished one is");
}

/// A query abandoned before its first row, while the server is still working on it, is
/// cancelled, and its connection is not reused.
#[rstest]
#[test_log::test(tokio::test)]
async fn test_postgres_a_query_abandoned_before_its_first_row_is_cancelled(
    container_manager: &Mutex<ContainerManager>,
) {
    let mut container_manager = container_manager.lock().await;
    if !container_manager.claimed {
        container_manager.claimed = true;
        start_container(&mut container_manager).await;
    }

    let pool = single_connection_pool(container_manager.port).await;
    let before = backend_pid(&pool).await;

    let abandoned = tokio::time::timeout(
        std::time::Duration::from_secs(1),
        get_stream(
            Arc::clone(&pool) as Arc<DynPostgresConnectionPool>,
            "SELECT true AS slept FROM pg_sleep(60) AS before_its_first_row".to_string(),
            Arc::new(Schema::new(vec![Field::new(
                "slept",
                DataType::Boolean,
                true,
            )])),
        ),
    )
    .await;
    assert!(abandoned.is_err(), "the query has no row to return yet");

    let admin = common::get_postgres_connection_pool(container_manager.port)
        .await
        .expect("Postgres connection pool should be created")
        .connect_direct()
        .await
        .expect("Connection should be established");
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        let running: i64 = admin
            .conn
            .query_one(
                "SELECT count(*) FROM pg_stat_activity \
                 WHERE state = 'active' AND query LIKE '%before_its_first_row%' \
                 AND pid <> pg_backend_pid()",
                &[],
            )
            .await
            .expect("pg_stat_activity should be readable")
            .get(0);
        if running == 0 {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the abandoned query is still running"
        );
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    assert_ne!(
        backend_pid(&pool).await,
        before,
        "the connection the cancel was sent on is not reused"
    );
}
