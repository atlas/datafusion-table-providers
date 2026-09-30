use std::any::Any;
use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;

use crate::arrow_sql_gen::nested;
use crate::arrow_sql_gen::rows_to_arrow;
use crate::arrow_sql_gen::schema::pg_data_type_to_arrow_type;
use crate::arrow_sql_gen::schema::ParseContext;
use crate::pool::ConnectionManager;
use crate::tls;
use arrow::datatypes::Field;
use arrow::datatypes::Schema;
use arrow::datatypes::SchemaRef;
use arrow_schema::DataType;
use async_stream::stream;
use bb8_postgres::tokio_postgres::types::ToSql;
use datafusion_table_providers_common::util::handle_unsupported_type_error;
use datafusion_table_providers_common::util::schema::SchemaValidator;
use datafusion_table_providers_common::UnsupportedTypeAction;
use datafusion_table_providers_common::{SOURCE_BASE_TYPE_METADATA_KEY, SOURCE_TYPE_METADATA_KEY};

fn maybe_db_source_err(err: tokio_postgres::Error) -> Box<dyn Error + Send + Sync> {
    if let Some(err) = err.as_db_error() {
        Box::new(err.clone())
    } else {
        Box::new(err)
    }
}

/// A pooled Postgres connection obtained from a [`PostgresConnectionPool`](crate::pool::PostgresConnectionPool).
///
/// Dereferences to [`tokio_postgres::Client`](bb8_postgres::tokio_postgres::Client) for executing queries.
// Defined here rather than in `postgrespool` to avoid a type-resolution cycle
// between the two modules (postgrespool imports PostgresConnection, which uses
// this alias).
pub type PostgresPooledConnection = bb8::PooledConnection<'static, ConnectionManager>;
use datafusion::error::DataFusionError;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::sql::TableReference;
use futures::stream;
use futures::Stream;
use futures::StreamExt;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use snafu::prelude::*;
use tokio_postgres::Row;

use datafusion_table_providers_common::sql::db_connection_pool::dbconnection::AsyncDbConnection;
use datafusion_table_providers_common::sql::db_connection_pool::dbconnection::DbConnection;
use datafusion_table_providers_common::sql::db_connection_pool::dbconnection::Result;

/// Resolves every domain to the type it is ultimately over, as `domain_base(domain, base,
/// typmod)`. A domain's values are its base type's, so everything describes a domain by its
/// base. The modifier (`numeric(10,2)`, `varchar(8)`) is the innermost domain's: a domain
/// cannot take one of its own.
macro_rules! domain_base_cte {
    () => {
        r"
WITH RECURSIVE domain_chain AS (
    SELECT t.oid AS domain, t.typbasetype AS base, t.typtypmod AS typmod
    FROM pg_type t
    WHERE t.typtype = 'd'
UNION ALL
    SELECT c.domain, t.typbasetype, t.typtypmod
    FROM domain_chain c
    JOIN pg_type t ON t.oid = c.base
    WHERE t.typtype = 'd'
),
domain_base AS (
    SELECT c.domain, c.base, c.typmod
    FROM domain_chain c
    JOIN pg_type t ON t.oid = c.base
    WHERE t.typtype <> 'd'
)
"
    };
}

const SCHEMA_QUERY: &str = concat!(
    domain_base_cte!(),
    r",
columns AS (
    SELECT
        a.attnum,
        a.attname,
        a.attnotnull,
        a.atttypid AS declared_type,
        a.atttypmod AS declared_typmod,
        COALESCE(db.base, a.atttypid) AS type,
        COALESCE(db.typmod, a.atttypmod) AS typmod
    FROM pg_class cls
    JOIN pg_namespace ns ON cls.relnamespace = ns.oid
    JOIN pg_attribute a ON a.attrelid = cls.oid
    LEFT JOIN domain_base db ON db.domain = a.atttypid
    WHERE ns.nspname = $1
        AND cls.relname = $2
        AND cls.relkind IN ('r','v','m','f','p')  -- covers tables, normal views, materialized views, foreign tables, & partitioned tables
        AND a.attnum > 0
        AND NOT a.attisdropped
)
SELECT
    c.attname AS column_name,
    CASE
    -- when an array type is encountered, label as 'array'
    WHEN t.typcategory = 'A' THEN 'array'
    -- if it’s a user-defined enum or composite type then output that specific string
    WHEN t.typtype = 'e' THEN 'enum'
    WHEN t.typtype = 'c' THEN 'composite'
    ELSE pg_catalog.format_type(c.type, c.typmod)
    END AS data_type,
    -- The type as Postgres itself names it, uncategorized: the branches above replace
    -- `text[]`/`mood` with the 'array'/'enum' labels the Arrow mapping dispatches on,
    -- which loses exactly the types the Arrow mapping is lossiest about. A domain is named
    -- as itself here, not as its base.
    pg_catalog.format_type(c.declared_type, c.declared_typmod) AS source_type,
    CASE WHEN c.attnotnull THEN 'NO' ELSE 'YES' END AS is_nullable,
    CASE
    WHEN t.typcategory = 'A' THEN
        jsonb_build_object(
        'type', 'array',
        'element_type', pg_catalog.format_type(
            et.oid,
            CASE WHEN edb.domain IS NULL THEN c.typmod ELSE edb.typmod END
        ),
        -- When the array element is a composite type, carry its attributes so the
        -- array can be resolved to a List<Struct>. Null for non-composite elements.
        'element_details', CASE WHEN et.typtype = 'c' THEN
            jsonb_build_object(
                'type', 'composite',
                'attributes', (
                    SELECT jsonb_agg(
                        jsonb_build_object(
                            'name', a2.attname,
                            'type', pg_catalog.format_type(
                                COALESCE(adb.base, a2.atttypid),
                                COALESCE(adb.typmod, a2.atttypmod)
                            )
                        )
                        ORDER BY a2.attnum
                    )
                    FROM pg_attribute a2
                    LEFT JOIN domain_base adb ON adb.domain = a2.atttypid
                    WHERE a2.attrelid = et.typrelid
                    AND a2.attnum > 0
                    AND NOT a2.attisdropped
                )
            )
        END
        )
    WHEN t.typtype = 'e' THEN
        jsonb_build_object(
            'type', 'enum',
            'values', (
                SELECT jsonb_agg(e.enumlabel ORDER BY e.enumsortorder)
                FROM pg_enum e
                WHERE e.enumtypid = t.oid
            )
        )
    WHEN t.typtype = 'c' THEN
        jsonb_build_object(
            'type', 'composite',
            'attributes', (
                SELECT jsonb_agg(
                    jsonb_build_object(
                        'name', a2.attname,
                        'type', pg_catalog.format_type(
                            COALESCE(adb.base, a2.atttypid),
                            COALESCE(adb.typmod, a2.atttypmod)
                        )
                    )
                    ORDER BY a2.attnum
                )
                FROM pg_attribute a2
                LEFT JOIN domain_base adb ON adb.domain = a2.atttypid
                WHERE a2.attrelid = t.typrelid
                AND a2.attnum > 0
                AND NOT a2.attisdropped
            )
        )
    END AS type_details,
    c.typmod,
    -- For a domain, the type it is over, formatted as a column of that type would be.
    CASE WHEN c.declared_type <> c.type
        THEN pg_catalog.format_type(c.type, c.typmod)
    END AS source_base_type
FROM columns c
JOIN pg_type t ON t.oid = c.type
-- An array's element, itself read as its base if it is a domain.
LEFT JOIN domain_base edb ON edb.domain = t.typelem
LEFT JOIN pg_type et ON et.oid = COALESCE(edb.base, NULLIF(t.typelem, 0))
ORDER BY c.attnum;
"
);

/// The type modifier of every live attribute of the composite types in `$1`, through any
/// domain, by type and position. See `nested::Modifiers`.
const ATTRIBUTE_MODIFIERS_QUERY: &str = concat!(
    domain_base_cte!(),
    r"
SELECT
    t.oid,
    row_number() OVER (PARTITION BY t.oid ORDER BY a.attnum) - 1 AS position,
    COALESCE(db.typmod, a.atttypmod) AS typmod
FROM pg_type t
JOIN pg_attribute a ON a.attrelid = t.typrelid
LEFT JOIN domain_base db ON db.domain = a.atttypid
WHERE t.oid = ANY($1)
    AND a.attnum > 0
    AND NOT a.attisdropped
"
);

// Redshift schema inference uses `SHOW COLUMNS FROM TABLE <db>.<schema>.<table>` rather
// than the `svv_*` catalog views. The catalog views truncate `data_type` at the source
// (`svv_all_columns.data_type` is `varchar(128)`; `svv_external_columns.external_type` is
// `text`, i.e. `varchar(256)` on Redshift), which corrupts deeply nested Spectrum complex
// types (`array<struct<...>>`) — and casting can't recover characters already dropped
// inside the view. `SHOW COLUMNS` returns the full, untruncated type.
//
// `SHOW COLUMNS` is a standalone command: it can't take bind parameters or be wrapped in
// a CTE/subquery, so we interpolate quoted identifiers (see `quote_pg_identifier`) and do
// the `numeric(p,s)` rebuild and `is_nullable` normalization in Rust instead of SQL (see
// `redshift_columns`). It covers local, datashare, and external tables.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostgresVariant {
    Default,
    Redshift,
}

/// A column description normalized across the per-variant schema-inference queries, so
/// `get_schema` can build Arrow fields without caring how the metadata was obtained.
struct ColumnDef {
    name: String,
    /// Fully formatted SQL type string (e.g. `numeric(10,2)`, `array<struct<...>>`).
    data_type: String,
    /// The source type as the database names it, before any categorization the Arrow
    /// mapping needs.
    source_type: String,
    nullable: bool,
    type_details: Option<serde_json::Value>,
    /// The type modifier (`atttypmod`, through any domain); -1 when there is none or it is
    /// not known.
    typmod: i32,
    /// For a domain, the type it is ultimately over. `None` for any other column.
    source_base_type: Option<String>,
}

impl ColumnDef {
    /// The Arrow field metadata recording the column's type as the source names it: the
    /// Arrow mapping is lossy (`varchar(50)` and `citext` both land on `Utf8`).
    fn metadata(&self) -> HashMap<String, String> {
        let mut metadata = HashMap::from([(
            SOURCE_TYPE_METADATA_KEY.to_string(),
            self.source_type.clone(),
        )]);
        if let Some(base) = &self.source_base_type {
            metadata.insert(SOURCE_BASE_TYPE_METADATA_KEY.to_string(), base.clone());
        }
        metadata
    }
}

/// Quotes a SQL identifier for safe interpolation into a statement that can't use bind
/// parameters (e.g. Redshift `SHOW COLUMNS`), doubling any embedded double quotes.
fn quote_pg_identifier(ident: &str) -> String {
    format!("\"{}\"", ident.replace('"', "\"\""))
}

/// Reads an optional integer column without assuming its wire integer width
/// (`SHOW COLUMNS` may report `numeric_precision`/`numeric_scale` as int2/int4/int8).
fn row_opt_int(row: &Row, column: &str) -> Option<i64> {
    if let Ok(v) = row.try_get::<_, Option<i32>>(column) {
        return v.map(i64::from);
    }
    if let Ok(v) = row.try_get::<_, Option<i64>>(column) {
        return v;
    }
    if let Ok(v) = row.try_get::<_, Option<i16>>(column) {
        return v.map(i64::from);
    }
    None
}

/// Maps an error from the catalog schema query into the right `datafusion_table_providers_common::sql::db_connection_pool::dbconnection::Error`, surfacing a
/// missing relation as [`datafusion_table_providers_common::sql::db_connection_pool::dbconnection::Error::UndefinedTable`].
fn map_schema_query_error(
    e: tokio_postgres::Error,
    table_reference: &TableReference,
) -> datafusion_table_providers_common::sql::db_connection_pool::dbconnection::Error {
    if let Some(db_error) = e.as_db_error() {
        if db_error.code() == &tokio_postgres::error::SqlState::UNDEFINED_TABLE {
            return datafusion_table_providers_common::sql::db_connection_pool::dbconnection::Error::UndefinedTable {
                source: Box::new(db_error.clone()),
                table_name: table_reference.to_string(),
            };
        }
    }
    datafusion_table_providers_common::sql::db_connection_pool::dbconnection::Error::UnableToGetSchema {
        source: maybe_db_source_err(e),
    }
}

const SCHEMAS_QUERY: &str = "
SELECT nspname AS schema_name
FROM pg_namespace
WHERE nspname NOT IN ('pg_catalog', 'information_schema')
  AND nspname !~ '^pg_toast';
";

const TABLES_QUERY: &str = "
SELECT tablename
FROM pg_tables
WHERE schemaname = $1;
";

// Schema/table discovery reads from the unified `svv_all_*` system views so that
// schemas/tables consumed from a datashare (absent from the local `pg_catalog`) and
// external schemas/tables (Redshift Spectrum) are listed alongside the connected
// database's own schemas/tables. `svv_all_schemas`/`svv_all_tables` are AWS's unions of
// the `svv_redshift_*` views with the external schemas/tables. `current_database()`
// scopes results to the connected (datashare consumer) database. (Per-column type
// inference, by contrast, uses `SHOW COLUMNS` — see `redshift_columns` — because the
// `svv_*` views truncate `data_type`.)
const REDSHIFT_SCHEMAS_QUERY: &str = "
SELECT schema_name
FROM svv_all_schemas
WHERE database_name = current_database()
  AND schema_name NOT IN ('pg_catalog', 'information_schema')
  AND schema_name NOT LIKE 'pg_temp_%'
  AND schema_name NOT LIKE 'pg_toast%';
";

const REDSHIFT_TABLES_QUERY: &str = "
SELECT table_name
FROM svv_all_tables
WHERE database_name = current_database()
  AND schema_name = $1;
";

#[derive(Debug, Snafu)]
pub enum PostgresError {
    #[snafu(display(
        "Query execution failed.\n{}\nFor details, refer to the PostgreSQL manual: https://www.postgresql.org/docs/17/index.html",
        format_postgres_query_error(source)
    ))]
    QueryError {
        source: bb8_postgres::tokio_postgres::Error,
    },

    #[snafu(display("Failed to convert query result to Arrow.\n{source}\nReport a bug to request support: https://github.com/datafusion-contrib/datafusion-table-providers/issues"))]
    ConversionError { source: crate::arrow_sql_gen::Error },
}

fn format_postgres_query_error(source: &bb8_postgres::tokio_postgres::Error) -> String {
    let Some(db_error) = source.as_db_error() else {
        return source.to_string();
    };

    let mut rendered = format!("{db_error}\nSQLSTATE: {}", db_error.code().code());

    let context: Vec<_> = [
        ("schema", db_error.schema()),
        ("table", db_error.table()),
        ("column", db_error.column()),
    ]
    .into_iter()
    .filter_map(|(k, v)| v.map(|v| format!("{k}={v}")))
    .collect();

    if !context.is_empty() {
        rendered.push_str("\nCONTEXT: ");
        rendered.push_str(&context.join(", "));
    }

    rendered
}

/// How long a request to cancel a query may take before it is abandoned.
const CANCEL_TIMEOUT: Duration = Duration::from_secs(5);

/// A query's rows that, dropped before they end, have the server stop producing them rather
/// than run the query to completion, or its timeout, for no one.
struct CancelOnDrop<S, F: FnOnce()> {
    rows: S,
    cancel: Option<F>,
}

impl<S, F: FnOnce()> CancelOnDrop<S, F> {
    fn new(rows: S, cancel: F) -> Self {
        Self {
            rows,
            cancel: Some(cancel),
        }
    }
}

impl<S, T, E, F> Stream for CancelOnDrop<S, F>
where
    S: Stream<Item = std::result::Result<T, E>> + Unpin,
    F: FnOnce() + Unpin,
{
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let polled = self.rows.poll_next_unpin(cx);
        // After its end or an error, the statement is no longer running.
        if matches!(polled, Poll::Ready(None | Some(Err(_)))) {
            self.cancel = None;
        }
        polled
    }
}

impl<S, F: FnOnce()> Drop for CancelOnDrop<S, F> {
    fn drop(&mut self) {
        if let Some(cancel) = self.cancel.take() {
            cancel();
        }
    }
}

pub struct PostgresConnection {
    pub conn: PostgresPooledConnection,
    unsupported_type_action: UnsupportedTypeAction,
    /// The TLS a request to cancel a query connects with; without it, none is sent.
    cancel_tls: Option<tls::Connector>,
}

impl SchemaValidator for PostgresConnection {
    type Error = datafusion_table_providers_common::sql::db_connection_pool::dbconnection::Error;

    fn is_data_type_supported(data_type: &DataType) -> bool {
        !matches!(data_type, DataType::Map(_, _))
    }

    fn unsupported_type_error(data_type: &DataType, field_name: &str) -> Self::Error {
        datafusion_table_providers_common::sql::db_connection_pool::dbconnection::Error::UnsupportedDataType {
            data_type: data_type.to_string(),
            field_name: field_name.to_string(),
        }
    }
}

impl<'a> DbConnection<PostgresPooledConnection, &'a (dyn ToSql + Sync)> for PostgresConnection {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn as_async(
        &self,
    ) -> Option<&dyn AsyncDbConnection<PostgresPooledConnection, &'a (dyn ToSql + Sync)>> {
        Some(self)
    }
}

#[async_trait::async_trait]
impl<'a> AsyncDbConnection<PostgresPooledConnection, &'a (dyn ToSql + Sync)>
    for PostgresConnection
{
    fn new(conn: PostgresPooledConnection) -> Self {
        PostgresConnection {
            conn,
            unsupported_type_action: UnsupportedTypeAction::default(),
            cancel_tls: None,
        }
    }

    async fn tables(
        &self,
        schema: &str,
    ) -> Result<
        Vec<String>,
        datafusion_table_providers_common::sql::db_connection_pool::dbconnection::Error,
    > {
        let query = match self.get_variant().await? {
            PostgresVariant::Default => TABLES_QUERY,
            PostgresVariant::Redshift => REDSHIFT_TABLES_QUERY,
        };

        let rows = self.conn.query(query, &[&schema]).await.map_err(|e| {
            datafusion_table_providers_common::sql::db_connection_pool::dbconnection::Error::UnableToGetTables {
                source: maybe_db_source_err(e),
            }
        })?;

        Ok(rows.iter().map(|r| r.get::<usize, String>(0)).collect())
    }

    async fn schemas(
        &self,
    ) -> Result<
        Vec<String>,
        datafusion_table_providers_common::sql::db_connection_pool::dbconnection::Error,
    > {
        let query = match self.get_variant().await? {
            PostgresVariant::Default => SCHEMAS_QUERY,
            PostgresVariant::Redshift => REDSHIFT_SCHEMAS_QUERY,
        };

        let rows =
            self.conn
                .query(query, &[])
                .await
                .map_err(|e| datafusion_table_providers_common::sql::db_connection_pool::dbconnection::Error::UnableToGetSchemas {
                    source: maybe_db_source_err(e),
                })?;

        Ok(rows.iter().map(|r| r.get::<usize, String>(0)).collect())
    }

    async fn get_schema(
        &self,
        table_reference: &TableReference,
    ) -> Result<
        SchemaRef,
        datafusion_table_providers_common::sql::db_connection_pool::dbconnection::Error,
    > {
        let (variant, columns) = self.query_variant_and_schema(table_reference).await?;

        // Native inference can return zero rows even though the table exists and is
        // queryable — e.g. Redshift datashare objects that aren't represented in the
        // catalog views we query. Fall back to inferring the schema from a sample row.
        if columns.is_empty() {
            tracing::warn!(
                "Native PostgreSQL schema inference returned no data. Inferring schema from table data rows."
            );
            return self.infer_schema_from_data(table_reference).await;
        }

        let (nested_types, modifiers) = match variant {
            PostgresVariant::Default => self.nested_types(table_reference, &columns).await?,
            PostgresVariant::Redshift => (HashMap::new(), nested::Modifiers::new()),
        };

        let mut fields = Vec::new();
        for column in columns {
            // Typed by the same mapping its rows are decoded with; see `nested`.
            if let Some(ty) = nested_types.get(&column.name) {
                match nested::data_type_with(ty, column.typmod, &modifiers, &column.name) {
                    Ok(arrow_type) => fields.push(
                        Field::new(&column.name, arrow_type, column.nullable)
                            .with_metadata(column.metadata()),
                    ),
                    Err(_) => handle_unsupported_type_error(
                        self.unsupported_type_action,
                        datafusion_table_providers_common::sql::db_connection_pool::dbconnection::Error::UnsupportedDataType {
                            data_type: column.source_type.clone(),
                            field_name: column.name.clone(),
                        },
                    )?,
                }
                continue;
            }

            let mut context =
                ParseContext::new().with_unsupported_type_action(self.unsupported_type_action);

            if let Some(type_details) = &column.type_details {
                context = context.with_type_details(type_details.clone());
            };

            let Ok(arrow_type) =
                pg_data_type_to_arrow_type(&column.data_type, &context, Some(variant))
            else {
                handle_unsupported_type_error(
                    self.unsupported_type_action,
                    datafusion_table_providers_common::sql::db_connection_pool::dbconnection::Error::UnsupportedDataType {
                        data_type: column.data_type.clone(),
                        field_name: column.name.clone(),
                    },
                )?;

                continue;
            };

            fields.push(
                Field::new(&column.name, arrow_type, column.nullable)
                    .with_metadata(column.metadata()),
            );
        }

        let schema = Arc::new(Schema::new(fields));
        Ok(schema)
    }

    async fn query_arrow(
        &self,
        sql: &str,
        params: &[&'a (dyn ToSql + Sync)],
        projected_schema: Option<SchemaRef>,
    ) -> Result<SendableRecordBatchStream> {
        // TODO: We should have a way to detect if params have been passed
        // if they haven't we should use .copy_out instead, because it should be much faster
        let cancel = self.canceller();
        let streamable = self
            .conn
            .query_raw(sql, params.iter().copied()) // use .query_raw to get access to the underlying RowStream
            .await
            .context(QuerySnafu)?;
        let streamable = CancelOnDrop::new(Box::pin(streamable), cancel);

        // chunk the stream into groups of rows
        let mut stream = streamable.chunks(4_000).boxed().map(move |rows| {
            let rows = rows
                .into_iter()
                .collect::<std::result::Result<Vec<_>, _>>()
                .context(QuerySnafu)?;
            let rec = rows_to_arrow(rows.as_slice(), &projected_schema).context(ConversionSnafu)?;
            Ok::<_, PostgresError>(rec)
        });

        let Some(first_chunk) = stream.next().await else {
            return Ok(Box::pin(RecordBatchStreamAdapter::new(
                Arc::new(Schema::empty()),
                stream::empty(),
            )));
        };

        let first_chunk = first_chunk?;
        let schema = first_chunk.schema(); // pull out the schema from the first chunk to use in the DataFusion Stream Adapter

        let output_stream = stream! {
           yield Ok(first_chunk);
           while let Some(batch) = stream.next().await {
                match batch {
                    Ok(batch) => {
                        yield Ok(batch); // we can yield the batch as-is because we've already converted to Arrow in the chunk map
                    }
                    Err(e) => {
                        yield Err(DataFusionError::Execution(format!("Failed to fetch batch: {e}")));
                    }
                }
           }
        };

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            output_stream,
        )))
    }

    async fn execute(&self, sql: &str, params: &[&'a (dyn ToSql + Sync)]) -> Result<u64> {
        Ok(self.conn.execute(sql, params).await?)
    }
}

impl PostgresConnection {
    #[must_use]
    pub fn with_unsupported_type_action(mut self, action: UnsupportedTypeAction) -> Self {
        self.unsupported_type_action = action;
        self
    }

    #[must_use]
    pub(crate) fn with_cancel_tls(mut self, tls: tls::Connector) -> Self {
        self.cancel_tls = Some(tls);
        self
    }

    /// Asks the server to cancel the query this connection is running, from a task of its
    /// own that gives up after [`CANCEL_TIMEOUT`]. Taken before the query is sent.
    #[cfg_attr(
        not(any(feature = "native-tls", feature = "rustls")),
        allow(clippy::clone_on_copy)
    )]
    fn canceller(&self) -> impl FnOnce() + Send + Unpin + 'static {
        let token = self.conn.cancel_token();
        let tls = self.cancel_tls.clone();
        let runtime = tokio::runtime::Handle::try_current().ok();
        move || {
            let (Some(tls), Some(runtime)) = (tls, runtime) else {
                return;
            };
            runtime.spawn(async move {
                match tokio::time::timeout(CANCEL_TIMEOUT, token.cancel_query(tls)).await {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => tracing::debug!("could not cancel a postgres query: {e}"),
                    Err(_) => tracing::debug!("cancelling a postgres query timed out"),
                }
            });
        }
    }

    pub async fn get_variant(
        &self,
    ) -> Result<
        PostgresVariant,
        datafusion_table_providers_common::sql::db_connection_pool::dbconnection::Error,
    > {
        let row = self
            .conn
            .query_one("SELECT version()", &[])
            .await
            .map_err(|e| datafusion_table_providers_common::sql::db_connection_pool::dbconnection::Error::UnableToGetSchema {
                source: maybe_db_source_err(e),
            })?;

        let version: String = row
            .try_get(0)
            .map_err(|e| datafusion_table_providers_common::sql::db_connection_pool::dbconnection::Error::UnableToGetSchema {
                source: maybe_db_source_err(e),
            })?;

        let variant = if version.contains("Redshift") {
            PostgresVariant::Redshift
        } else {
            PostgresVariant::Default
        };

        Ok(variant)
    }

    /// The Postgres types of those of `columns` that are decoded recursively — composites,
    /// and arrays of anything but the common scalars — keyed by column name, with the type
    /// modifiers of the attributes of every composite they hold.
    ///
    /// The catalog query describes a composite's attributes by type name only, which cannot
    /// say what a nested composite holds. The server describes each column's type in full
    /// for a statement, so these are read from one: prepared, never executed. A `Type`
    /// carries no modifiers, so those are read from the catalog for the composites found.
    async fn nested_types(
        &self,
        table_reference: &TableReference,
        columns: &[ColumnDef],
    ) -> Result<
        (
            HashMap<String, tokio_postgres::types::Type>,
            nested::Modifiers,
        ),
        datafusion_table_providers_common::sql::db_connection_pool::dbconnection::Error,
    > {
        let schema_error = |e| {
            datafusion_table_providers_common::sql::db_connection_pool::dbconnection::Error::UnableToGetSchema {
                source: maybe_db_source_err(e),
            }
        };

        let candidates: Vec<String> = columns
            .iter()
            .filter(|column| matches!(column.data_type.as_str(), "composite" | "array"))
            .map(|column| quote_pg_identifier(&column.name))
            .collect();
        if candidates.is_empty() {
            return Ok((HashMap::new(), nested::Modifiers::new()));
        }

        let sql = format!(
            "SELECT {} FROM {}.{}",
            candidates.join(", "),
            quote_pg_identifier(table_reference.schema().unwrap_or("public")),
            quote_pg_identifier(table_reference.table()),
        );
        let statement = self.conn.prepare(&sql).await.map_err(schema_error)?;
        let types: HashMap<_, _> = statement
            .columns()
            .iter()
            .filter(|column| nested::decodes(column.type_()))
            .map(|column| (column.name().to_string(), column.type_().clone()))
            .collect();

        let mut composites = Vec::new();
        for ty in types.values() {
            nested::composites(ty, &mut composites);
        }
        let mut modifiers = nested::Modifiers::new();
        if !composites.is_empty() {
            let rows = self
                .conn
                .query(ATTRIBUTE_MODIFIERS_QUERY, &[&composites])
                .await
                .map_err(schema_error)?;
            for row in rows {
                let position = usize::try_from(row.get::<usize, i64>(1)).unwrap_or_default();
                modifiers.insert(
                    (row.get::<usize, u32>(0), position),
                    row.get::<usize, i32>(2),
                );
            }
        }

        Ok((types, modifiers))
    }

    async fn query_variant_and_schema(
        &self,
        table_reference: &TableReference,
    ) -> Result<
        (PostgresVariant, Vec<ColumnDef>),
        datafusion_table_providers_common::sql::db_connection_pool::dbconnection::Error,
    > {
        let table_name = table_reference.table();
        let schema_name = table_reference.schema().unwrap_or("public");

        let variant = self.get_variant().await?;

        let columns = match variant {
            PostgresVariant::Default => {
                let rows = self
                    .conn
                    .query(SCHEMA_QUERY, &[&schema_name, &table_name])
                    .await
                    .map_err(|e| map_schema_query_error(e, table_reference))?;

                rows.iter()
                    .map(|row| ColumnDef {
                        name: row.get::<usize, String>(0),
                        // `data_type` is the categorized column: a `format_type`
                        // string (e.g. `numeric(10,2)`) except for arrays, enums, and
                        // composites, which the query rewrites to the labels
                        // `pg_data_type_to_arrow_type` dispatches on.
                        data_type: row.get::<usize, String>(1),
                        source_type: row.get::<usize, String>(2),
                        nullable: row.get::<usize, String>(3) == "YES",
                        type_details: row.get::<usize, Option<serde_json::Value>>(4),
                        typmod: row.get::<usize, i32>(5),
                        source_base_type: row.get::<usize, Option<String>>(6),
                    })
                    .collect()
            }
            PostgresVariant::Redshift => {
                self.redshift_columns(table_reference.catalog(), schema_name, table_name)
                    .await?
            }
        };

        Ok((variant, columns))
    }

    /// Infers a Redshift table's columns via `SHOW COLUMNS FROM TABLE`, which (unlike the
    /// `svv_*` catalog views) returns the full, untruncated `data_type` — essential for
    /// deeply nested Spectrum complex types. It can't take bind parameters or be composed,
    /// so identifiers are quoted/interpolated and the `numeric(p,s)` rebuild + nullable
    /// normalization happen here in Rust rather than in SQL.
    async fn redshift_columns(
        &self,
        catalog: Option<&str>,
        schema_name: &str,
        table_name: &str,
    ) -> Result<
        Vec<ColumnDef>,
        datafusion_table_providers_common::sql::db_connection_pool::dbconnection::Error,
    > {
        // `SHOW COLUMNS` needs a fully-qualified 3-part name including the database. Use
        // the catalog from the table reference when the caller fully-qualified it;
        // otherwise scope to the connected database (matching the previous
        // `current_database()` filter).
        let database_name: String = match catalog {
            Some(catalog) => catalog.to_string(),
            None => self
                .conn
                .query_one("SELECT current_database()", &[])
                .await
                .map_err(|e| datafusion_table_providers_common::sql::db_connection_pool::dbconnection::Error::UnableToGetSchema {
                    source: maybe_db_source_err(e),
                })?
                .try_get(0)
                .map_err(|e| datafusion_table_providers_common::sql::db_connection_pool::dbconnection::Error::UnableToGetSchema {
                    source: maybe_db_source_err(e),
                })?,
        };

        let sql = format!(
            "SHOW COLUMNS FROM TABLE {}.{}.{}",
            quote_pg_identifier(&database_name),
            quote_pg_identifier(schema_name),
            quote_pg_identifier(table_name),
        );

        let rows = match self.conn.query(&sql, &[]).await {
            Ok(rows) => rows,
            // `SHOW COLUMNS` raises `UNDEFINED_TABLE` for relations it can't resolve in the
            // target database (e.g. cross-database datashare objects). Treat only that as a
            // miss and fall back to data-based inference — matching the catalog path's
            // empty-result behavior — while surfacing any other error (permissions, syntax,
            // connectivity) instead of silently hiding it.
            Err(e)
                if e.as_db_error().map(tokio_postgres::error::DbError::code)
                    == Some(&tokio_postgres::error::SqlState::UNDEFINED_TABLE) =>
            {
                tracing::debug!(
                    "Redshift SHOW COLUMNS: {schema_name}.{table_name} not found in {database_name}; falling back to data inference"
                );
                return Ok(Vec::new());
            }
            Err(e) => {
                return Err(datafusion_table_providers_common::sql::db_connection_pool::dbconnection::Error::UnableToGetSchema {
                    source: maybe_db_source_err(e),
                })
            }
        };

        let columns = rows
            .iter()
            .map(|row| {
                let name: String = row.get("column_name");
                let mut data_type: String = row.get("data_type");

                // `SHOW COLUMNS` reports numeric/decimal as the bare type with precision
                // and scale in separate columns; rebuild the formatted `numeric(p,s)`.
                let lowered = data_type.to_lowercase();
                if lowered == "numeric" || lowered == "decimal" {
                    if let Some(precision) = row_opt_int(row, "numeric_precision") {
                        let scale = row_opt_int(row, "numeric_scale").unwrap_or(0);
                        data_type = format!("{data_type}({precision},{scale})");
                    }
                }

                // Normalize nullability: only an explicit no/false is non-nullable; an
                // empty/absent value (common for external columns) is treated as nullable.
                let nullable = row
                    .try_get::<_, Option<String>>("is_nullable")
                    .ok()
                    .flatten()
                    .map(|v| !matches!(v.to_lowercase().as_str(), "no" | "false"))
                    .unwrap_or(true);

                ColumnDef {
                    name,
                    // `SHOW COLUMNS` reports the full formatted type and this path does
                    // not categorize it, so it is already the source type.
                    source_type: data_type.clone(),
                    data_type,
                    nullable,
                    type_details: None,
                    typmod: -1,
                    source_base_type: None,
                }
            })
            .collect();

        Ok(columns)
    }

    /// Fallback schema inference used when native (catalog-based) inference returns no
    /// rows for a table that is nonetheless queryable. Reads a single row and derives the
    /// Arrow schema from its column metadata via `rows_to_arrow`.
    ///
    /// Note: an empty table yields an empty schema, since `rows_to_arrow` reads column
    /// types from the first returned row.
    async fn infer_schema_from_data(
        &self,
        table_reference: &TableReference,
    ) -> Result<
        SchemaRef,
        datafusion_table_providers_common::sql::db_connection_pool::dbconnection::Error,
    > {
        let rows = self
            .conn
            .query(&format!("SELECT * FROM {table_reference} LIMIT 1"), &[])
            .await
            .map_err(|e| {
                if let Some(error_source) = e.source() {
                    if let Some(pg_error) =
                        error_source.downcast_ref::<tokio_postgres::error::DbError>()
                    {
                        if pg_error.code() == &tokio_postgres::error::SqlState::UNDEFINED_TABLE {
                            return datafusion_table_providers_common::sql::db_connection_pool::dbconnection::Error::UndefinedTable {
                                source: Box::new(pg_error.clone()),
                                table_name: table_reference.to_string(),
                            };
                        }
                    }
                }
                datafusion_table_providers_common::sql::db_connection_pool::dbconnection::Error::UnableToGetSchema {
                    source: maybe_db_source_err(e),
                }
            })?;

        if rows.is_empty() {
            tracing::warn!(
                "Schema inference from table data rows returned no rows. The schema for table '{table_reference}' is empty.",
            );
            return Ok(Arc::new(Schema::empty()));
        }

        let rec =
            rows_to_arrow(rows.as_slice(), &None).map_err(|e| datafusion_table_providers_common::sql::db_connection_pool::dbconnection::Error::UnableToGetSchema {
                source: Box::new(PostgresError::ConversionError { source: e }),
            })?;

        Ok(rec.schema())
    }
}

#[cfg(test)]
mod tests {
    use super::CancelOnDrop;
    use futures::{stream, StreamExt};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    type Rows = stream::Iter<std::vec::IntoIter<Result<u8, ()>>>;

    fn rows(items: Vec<Result<u8, ()>>) -> (CancelOnDrop<Rows, impl FnOnce()>, Arc<AtomicBool>) {
        let cancelled = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&cancelled);
        let rows = CancelOnDrop::new(stream::iter(items), move || {
            flag.store(true, Ordering::SeqCst)
        });
        (rows, cancelled)
    }

    #[tokio::test]
    async fn a_stream_dropped_before_its_end_cancels_its_query() {
        let (mut unread, cancelled) = rows(vec![Ok(1), Ok(2)]);
        assert_eq!(unread.next().await, Some(Ok(1)));
        assert!(!cancelled.load(Ordering::SeqCst));
        drop(unread);
        assert!(cancelled.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn a_stream_that_ended_cancels_nothing() {
        let (read, cancelled) = rows(vec![Ok(1), Ok(2)]);
        assert_eq!(read.collect::<Vec<_>>().await, [Ok(1), Ok(2)]);
        assert!(!cancelled.load(Ordering::SeqCst));

        let (mut failed, cancelled) = rows(vec![Err(()), Ok(2)]);
        assert_eq!(failed.next().await, Some(Err(())));
        drop(failed);
        assert!(
            !cancelled.load(Ordering::SeqCst),
            "a statement that failed is no longer running"
        );
    }
}
