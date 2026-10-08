//! Writes through the table writer, which uses binary `COPY` for every column type it can
//! encode and an `INSERT` statement otherwise.
use std::sync::Arc;

use datafusion::common::Constraints;
use datafusion::execution::context::SessionContext;
use datafusion::sql::TableReference;
use datafusion_table_providers::postgres::{
    write::PostgresTableWriter, Postgres, PostgresTableFactory,
};
use datafusion_table_providers::util::{
    column_reference::ColumnReference, on_conflict::OnConflict,
};
use datafusion_table_providers::UnsupportedTypeAction;

use super::common;

/// Runs `statements` on the database, in order.
async fn execute(port: usize, statements: &str) {
    let pool = common::get_postgres_connection_pool(port)
        .await
        .expect("Postgres connection pool should be created");
    let conn = pool
        .connect_direct()
        .await
        .expect("Connection should be established");
    conn.conn
        .batch_execute(statements)
        .await
        .expect("Statements should run");
}

/// Each row of `query`, its columns as text.
async fn rows(port: usize, query: &str) -> Vec<Vec<Option<String>>> {
    let pool = common::get_postgres_connection_pool(port)
        .await
        .expect("Postgres connection pool should be created");
    let conn = pool
        .connect_direct()
        .await
        .expect("Connection should be established");
    conn.conn
        .query(query, &[])
        .await
        .expect("Query should run")
        .iter()
        .map(|row| (0..row.len()).map(|i| row.get(i)).collect())
        .collect()
}

/// A session where `table` is the writer over the table of that name.
async fn session(port: usize, table: &str, on_conflict: Option<OnConflict>) -> SessionContext {
    let pool = Arc::new(
        common::get_postgres_connection_pool(port)
            .await
            .expect("Postgres connection pool should be created")
            .with_unsupported_type_action(UnsupportedTypeAction::String),
    );
    let reference = TableReference::bare(table);
    let read = PostgresTableFactory::new(Arc::clone(&pool))
        .table_provider(reference.clone())
        .await
        .expect("Table should be readable");
    let schema = read.schema();
    let writer = PostgresTableWriter::create(
        read,
        Postgres::new(reference, pool, schema, Constraints::default()),
        on_conflict,
    );
    let ctx = SessionContext::new();
    ctx.register_table(table, writer)
        .expect("Table should be registered");
    ctx
}

async fn write(ctx: &SessionContext, statement: &str) {
    ctx.sql(statement)
        .await
        .expect("Statement should be planned")
        .collect()
        .await
        .expect("Rows should be written");
}

/// Composites, arrays of them, `jsonb`, enums, `numeric`, `uuid` and `timestamptz`, with
/// nulls at every level: the column, a composite's member and an array's element.
pub(super) async fn test_postgres_copy_writes_composites_arrays_and_jsonb(port: usize) {
    execute(
        port,
        "
        DROP TABLE IF EXISTS copy_values;
        DROP TYPE IF EXISTS copy_item;
        DROP TYPE IF EXISTS copy_mood;
        CREATE TYPE copy_mood AS ENUM ('happy', 'sad');
        CREATE TYPE copy_item AS (sku TEXT, qty INT, price DOUBLE PRECISION);
        CREATE TABLE copy_values (
            id INT PRIMARY KEY,
            key UUID,
            item copy_item,
            items copy_item[],
            data JSONB,
            mood copy_mood,
            price NUMERIC(10, 3),
            at TIMESTAMPTZ
        );",
    )
    .await;
    let ctx = session(port, "copy_values", None).await;
    write(
        &ctx,
        r#"INSERT INTO copy_values VALUES
            (1, '00000000-0000-0000-0000-00000000002a',
             named_struct('sku', 'a', 'qty', 2, 'price', 9.5),
             make_array(
                 named_struct('sku', 'a', 'qty', 2, 'price', 9.5),
                 named_struct('sku', CAST(NULL AS VARCHAR), 'qty', 1, 'price', 0.1)),
             '{"a": [1, null]}', 'happy', 12.345, '2024-02-29T12:34:56.123456Z'),
            (2, NULL, NULL, NULL, 'null', NULL, -0.5, NULL)"#,
    )
    .await;

    assert_eq!(
        rows(
            port,
            "SELECT id::text, key::text, item::text, items::text, data::text, mood::text,
                    price::text, extract(epoch FROM at)::text
             FROM copy_values ORDER BY id",
        )
        .await,
        vec![
            vec![
                Some("1".into()),
                Some("00000000-0000-0000-0000-00000000002a".into()),
                Some("(a,2,9.5)".into()),
                // A null member is empty; an empty string would be written "".
                Some(r#"{"(a,2,9.5)","(,1,0.1)"}"#.into()),
                Some(r#"{"a": [1, null]}"#.into()),
                Some("happy".into()),
                Some("12.345".into()),
                Some("1709210096.123456".into()),
            ],
            vec![
                Some("2".into()),
                None,
                None,
                None,
                Some("null".into()),
                None,
                Some("-0.500".into()),
                None,
            ],
        ]
    );
}

/// With `on_conflict`, rows are copied into a temporary table and inserted from it.
pub(super) async fn test_postgres_copy_resolves_conflicts(port: usize) {
    execute(
        port,
        "
        DROP TABLE IF EXISTS copy_conflicts;
        CREATE TABLE copy_conflicts (id INT PRIMARY KEY, name TEXT);",
    )
    .await;
    let id = || ColumnReference::try_from("id").expect("A column reference");
    let plain = session(port, "copy_conflicts", None).await;
    write(
        &plain,
        "INSERT INTO copy_conflicts VALUES (1, 'a'), (2, 'b')",
    )
    .await;
    let upsert = session(port, "copy_conflicts", Some(OnConflict::Upsert(id()))).await;
    write(
        &upsert,
        "INSERT INTO copy_conflicts VALUES (2, 'c'), (3, 'd')",
    )
    .await;
    let keep = session(port, "copy_conflicts", Some(OnConflict::DoNothing(id()))).await;
    write(
        &keep,
        "INSERT INTO copy_conflicts VALUES (3, 'e'), (4, 'f')",
    )
    .await;

    assert_eq!(
        rows(
            port,
            "SELECT id::text, name FROM copy_conflicts ORDER BY id"
        )
        .await,
        [("1", "a"), ("2", "c"), ("3", "d"), ("4", "f")]
            .map(|(id, name)| vec![Some(id.to_owned()), Some(name.to_owned())])
    );
}

/// A column type binary `COPY` does not encode is written with `INSERT` instead.
pub(super) async fn test_postgres_copy_falls_back_to_insert(port: usize) {
    execute(
        port,
        "
        DROP TABLE IF EXISTS copy_fallback;
        CREATE TABLE copy_fallback (id INT PRIMARY KEY, address INET);",
    )
    .await;
    let ctx = session(port, "copy_fallback", None).await;
    write(&ctx, "INSERT INTO copy_fallback VALUES (1, '10.0.0.1')").await;

    assert_eq!(
        rows(port, "SELECT id::text, address::text FROM copy_fallback").await,
        vec![vec![Some("1".to_owned()), Some("10.0.0.1/32".to_owned())]]
    );
}
