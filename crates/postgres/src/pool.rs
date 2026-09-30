// Every connector but `NoTls`, the one a build without TLS has, is `Clone` and not `Copy`.
#![cfg_attr(
    not(any(feature = "native-tls", feature = "rustls")),
    allow(clippy::clone_on_copy)
)]

use crate::conn::PostgresConnection;
use crate::tls;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, Ordering};
use std::{collections::HashMap, str::FromStr, sync::Arc};

use async_trait::async_trait;
use bb8::ErrorSink;
use bb8_postgres::tokio_postgres::{config::Host, types::ToSql, Config};
use datafusion_table_providers_common::{
    util::{self, ns_lookup::verify_ns_lookup_and_tcp_connect},
    UnsupportedTypeAction,
};
use secrecy::{ExposeSecret, SecretBox, SecretString};
use snafu::{prelude::*, ResultExt};
use tokio::runtime::Handle;
use tokio_postgres;

use datafusion_table_providers_common::sql::db_connection_pool::{
    dbconnection::{AsyncDbConnection, DbConnection},
    runtime::run_async_with_tokio,
    DbConnectionPool, JoinPushDown, PasswordProvider, StaticPasswordProvider,
};

#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum Error {
    #[snafu(display("PostgreSQL connection failed.\n{source}\nFor details, refer to the PostgreSQL documentation: https://www.postgresql.org/docs/17/index.html"))]
    ConnectionPoolError {
        source: bb8_postgres::tokio_postgres::Error,
    },

    #[snafu(display("PostgreSQL connection failed.\n{source}\nAdjust the connection pool parameters for sufficient capacity."))]
    ConnectionPoolRunError {
        source: bb8::RunError<bb8_postgres::tokio_postgres::Error>,
    },

    #[snafu(display(
        "Invalid parameter: {parameter_name}. Ensure the parameter name is correct."
    ))]
    InvalidParameterError { parameter_name: String },

    #[snafu(display("Could not parse {parameter_name} into a valid integer. Ensure it is configured with a valid value."))]
    InvalidIntegerParameterError {
        parameter_name: String,
        source: std::num::ParseIntError,
    },

    #[snafu(display("Cannot connect to PostgreSQL on {host}:{port}. Ensure the host and port are correct and reachable."))]
    InvalidHostOrPortError {
        source: datafusion_table_providers_common::util::ns_lookup::Error,
        host: String,
        port: u16,
    },

    #[snafu(display(
        "Invalid root certificate path: {path}. Ensure it points to a valid root certificate."
    ))]
    InvalidRootCertPathError { path: String },

    #[snafu(display(
        "sslrootcert and sslrootcert_pem are both set. Trust the roots in a file or in memory, not both."
    ))]
    ConflictingRootCertsError,

    #[snafu(display(
        "Invalid hostaddr. It must list one IP address for each host, separated by commas."
    ))]
    InvalidHostaddrError,

    #[snafu(display(
        "Failed to read certificate.\n{source}\nEnsure the root certificate path points to a valid certificate."
    ))]
    FailedToReadCertError { source: std::io::Error },

    #[snafu(display(
        "Certificate loading failed.\n{source}\nEnsure the root certificate path points to a valid certificate."
    ))]
    FailedToLoadCertError { source: BoxError },

    #[snafu(display("TLS connector initialization failed.\n{source}\nVerify SSL mode and root certificate validity"))]
    FailedToBuildTlsConnectorError { source: BoxError },

    #[snafu(display("sslmode={ssl_mode} needs TLS, which this build has none of. Enable the `native-tls` or `rustls` feature, or connect with sslmode=disable."))]
    TlsNotCompiled { ssl_mode: String },

    #[snafu(display("PostgreSQL connection failed.\n{source}\nFor details, refer to the PostgreSQL documentation: https://www.postgresql.org/docs/17/index.html"))]
    PostgresConnectionError { source: tokio_postgres::Error },

    #[snafu(display("Authentication failed. Verify username and password."))]
    InvalidUsernameOrPassword { source: tokio_postgres::Error },

    #[snafu(display("Password provider error.\n{source}"))]
    PasswordProviderError {
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[snafu(display("Task failed to execute on IO runtime.\n{source}"))]
    IoRuntimeError { source: tokio::task::JoinError },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// The error of a TLS implementation, which varies with the one compiled in.
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Error type for the connection manager, covering both Postgres and password provider errors.
#[derive(Debug)]
pub enum ConnectionManagerError {
    /// An error from the underlying Postgres connection.
    Postgres(tokio_postgres::Error),
    /// An error from the password provider.
    PasswordProvider(Box<dyn std::error::Error + Send + Sync>),
}

impl std::fmt::Display for ConnectionManagerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Postgres(e) => write!(f, "{e}"),
            Self::PasswordProvider(e) => write!(f, "password provider error: {e}"),
        }
    }
}

impl std::error::Error for ConnectionManagerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Postgres(e) => Some(e),
            Self::PasswordProvider(e) => Some(e.as_ref()),
        }
    }
}

impl From<tokio_postgres::Error> for ConnectionManagerError {
    fn from(e: tokio_postgres::Error) -> Self {
        Self::Postgres(e)
    }
}

/// A bb8 connection manager that supports dynamic password providers.
///
/// When a [`PasswordProvider`] is set, the manager calls it to get a fresh password
/// each time a new connection is created. This enables rotating credentials,
/// JWT-based auth, and cloud IAM authentication.
///
/// When no provider is set (passwordless auth), the manager connects using the
/// stored [`Config`] as-is.
pub struct ConnectionManager {
    config: Config,
    tls: tls::Connector,
    password_provider: Option<Arc<dyn PasswordProvider>>,
}

impl ConnectionManager {
    fn new(config: Config, tls: tls::Connector) -> Self {
        Self {
            config,
            tls,
            password_provider: None,
        }
    }

    fn with_password_provider(mut self, provider: Arc<dyn PasswordProvider>) -> Self {
        self.password_provider = Some(provider);
        self
    }
}

/// A pooled client. Once a request to cancel a query on it has been sent, the pool discards
/// it rather than hand it out again, since that request could reach a later statement.
pub struct PostgresClient {
    client: tokio_postgres::Client,
    pub(crate) cancelled: Arc<AtomicBool>,
}

impl Deref for PostgresClient {
    type Target = tokio_postgres::Client;

    fn deref(&self) -> &Self::Target {
        &self.client
    }
}

impl DerefMut for PostgresClient {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.client
    }
}

/// Applies per-connection session configuration after a connection is established.
///
/// Redshift surfaces Spectrum complex external columns (`ARRAY`/`STRUCT`/`MAP`) and
/// `SUPER` values as JSON text only when `json_serialization_enable` is on; otherwise
/// selecting such a column errors server-side. `json_serialization_parse_nested_strings`
/// additionally renders nested string fields that hold valid JSON inline (unescaped)
/// rather than as escaped string literals, so the decoded values match their schema.
///
/// These parameters only exist on Redshift, so rather than spend a `SELECT version()`
/// round-trip detecting the variant we apply them optimistically and treat vanilla
/// PostgreSQL's "unrecognized configuration parameter" error (`SQLSTATE 42704`) as a
/// no-op. The `SET`s run outside a transaction, so a failure leaves the connection
/// usable.
///
/// See <https://docs.aws.amazon.com/redshift/latest/dg/r_json_serialization_enable.html>
/// and <https://docs.aws.amazon.com/redshift/latest/dg/r_json_serialization_parse_nested_strings.html>.
async fn configure_session(
    client: &tokio_postgres::Client,
) -> std::result::Result<(), ConnectionManagerError> {
    if let Err(e) = client
        .batch_execute(
            "SET json_serialization_enable TO true; \
             SET json_serialization_parse_nested_strings TO true;",
        )
        .await
    {
        // Vanilla PostgreSQL rejects these unknown parameters with `undefined_object`
        // (42704); that just means "not Redshift", so ignore it. Anything else is real.
        if e.code() == Some(&tokio_postgres::error::SqlState::UNDEFINED_OBJECT) {
            return Ok(());
        }
        return Err(e.into());
    }

    Ok(())
}

impl bb8::ManageConnection for ConnectionManager {
    type Connection = PostgresClient;
    type Error = ConnectionManagerError;

    async fn connect(&self) -> std::result::Result<PostgresClient, ConnectionManagerError> {
        let (client, connection) = if let Some(provider) = &self.password_provider {
            let password = provider
                .get_password()
                .await
                .map_err(ConnectionManagerError::PasswordProvider)?;
            let mut config = self.config.clone();
            config.password(password.expose_secret());
            config.connect(self.tls.clone()).await?
        } else {
            self.config.connect(self.tls.clone()).await?
        };
        tokio::spawn(async move {
            if let Err(e) = connection.await {
                tracing::debug!("postgres connection error: {e}");
            }
        });

        configure_session(&client).await?;

        Ok(PostgresClient {
            client,
            cancelled: Arc::default(),
        })
    }

    async fn is_valid(
        &self,
        conn: &mut PostgresClient,
    ) -> std::result::Result<(), ConnectionManagerError> {
        conn.simple_query("").await.map(|_| ())?;
        Ok(())
    }

    fn has_broken(&self, conn: &mut PostgresClient) -> bool {
        conn.is_closed() || conn.cancelled.load(Ordering::Relaxed)
    }
}

pub struct PostgresConnectionPool {
    pool: Arc<bb8::Pool<ConnectionManager>>,
    /// The TLS the pool's connections are made with, which their cancel requests need too.
    tls: tls::Connector,
    join_push_down: JoinPushDown,
    unsupported_type_action: UnsupportedTypeAction,
    io_handle: Option<Handle>,
}

impl std::fmt::Debug for PostgresConnectionPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PostgresConnectionPool")
            .field("pool", &self.pool)
            .field("join_push_down", &self.join_push_down)
            .field("unsupported_type_action", &self.unsupported_type_action)
            .field("io_handle", &self.io_handle)
            .finish_non_exhaustive()
    }
}

impl PostgresConnectionPool {
    /// Creates a new instance of `PostgresConnectionPool`.
    ///
    /// If a `pass` parameter is present, it is wrapped in a [`StaticPasswordProvider`]
    /// internally. For dynamic credentials, use [`new_with_password_provider`](Self::new_with_password_provider).
    ///
    /// # Errors
    ///
    /// Returns an error if there is a problem creating the connection pool.
    pub async fn new(params: HashMap<String, SecretString>) -> Result<Self> {
        Self::new_inner(params, None).await
    }

    /// Creates a new instance of `PostgresConnectionPool` with a dynamic password provider.
    ///
    /// The password provider is called each time a new connection is created in the pool,
    /// enabling support for rotating credentials, JWT tokens, and cloud IAM authentication.
    ///
    /// Any `pass` parameter in `params` is ignored; the provider is used instead.
    ///
    /// # Errors
    ///
    /// Returns an error if there is a problem creating the connection pool.
    pub async fn new_with_password_provider(
        params: HashMap<String, SecretString>,
        password_provider: Arc<dyn PasswordProvider>,
    ) -> Result<Self> {
        Self::new_inner(params, Some(password_provider)).await
    }

    async fn new_inner(
        params: HashMap<String, SecretString>,
        password_provider: Option<Arc<dyn PasswordProvider>>,
    ) -> Result<Self> {
        // Remove the "pg_" prefix from the keys to keep backward compatibility
        let params = util::remove_prefix_from_hashmap_keys(params, "pg_");

        let (config, verify) = config_from(&params)?;
        if verify {
            verify_postgres_config(&config).await?;
        }

        let root_certs = root_certs(&params).await?;
        let connector = tls::connector(ssl_mode(&params)?.as_str(), root_certs.as_deref())?;

        // Resolve the password provider: use the caller's, wrap the static password,
        // or leave as None for passwordless auth (trust, cert, etc.).
        let password_provider = password_provider.or_else(|| {
            static_password(&params)
                .map(|pw| Arc::new(StaticPasswordProvider::new(pw)) as Arc<dyn PasswordProvider>)
        });

        // Test the connection
        if let Some(ref provider) = password_provider {
            let password = provider
                .get_password()
                .await
                .map_err(|source| Error::PasswordProviderError { source })?;
            let mut test_config = config.clone();
            test_config.password(password.expose_secret());
            test_connection(&test_config, connector.clone()).await?;
        } else {
            test_connection(&config, connector.clone()).await?;
        }

        let join_push_down = get_join_context(&config);

        let mut manager = ConnectionManager::new(config, connector.clone());
        if let Some(provider) = password_provider {
            manager = manager.with_password_provider(provider);
        }
        let error_sink = PostgresErrorSink::new();

        let mut connection_pool_size = 10; // The BB8 default is 10
        if let Some(pg_pool_size) = params
            .get("connection_pool_size")
            .map(SecretBox::expose_secret)
        {
            connection_pool_size = pg_pool_size.parse().context(InvalidIntegerParameterSnafu {
                parameter_name: "pool_size".to_string(),
            })?;
        }

        let pool = bb8::Pool::builder()
            .max_size(connection_pool_size)
            .error_sink(Box::new(error_sink))
            .build(manager)
            .await
            .map_err(map_pool_build_error)?;

        // Verify the pool by executing a simple query
        {
            let conn = pool.get().await.map_err(map_pool_run_error)?;
            conn.execute("SELECT 1", &[])
                .await
                .context(ConnectionPoolSnafu)?;
        }

        Ok(PostgresConnectionPool {
            pool: Arc::new(pool),
            tls: connector,
            join_push_down,
            unsupported_type_action: UnsupportedTypeAction::default(),
            io_handle: None,
        })
    }

    /// Specify the action to take when an invalid type is encountered.
    #[must_use]
    pub fn with_unsupported_type_action(mut self, action: UnsupportedTypeAction) -> Self {
        self.unsupported_type_action = action;
        self
    }

    /// Route all Postgres connection background tasks to a dedicated IO runtime.
    #[must_use]
    pub fn with_io_runtime(mut self, handle: Handle) -> Self {
        self.io_handle = Some(handle);
        self
    }

    /// Returns a direct connection to the underlying database.
    ///
    /// # Errors
    ///
    /// Returns an error if there is a problem creating the connection pool.
    pub async fn connect_direct(&self) -> Result<PostgresConnection> {
        let pool = Arc::clone(&self.pool);
        let conn = if let Some(handle) = &self.io_handle {
            handle
                .spawn(async move { pool.get_owned().await.map_err(map_pool_run_error) })
                .await
                .context(IoRuntimeSnafu)??
        } else {
            pool.get_owned().await.map_err(map_pool_run_error)?
        };
        Ok(PostgresConnection::new(conn).with_cancel_tls(self.tls.clone()))
    }
}

/// Parses a connection string into components, extracting `sslmode`, `sslrootcert`,
/// and `password` separately so they can be handled by the caller.
fn parse_connection_string(
    pg_connection_string: &str,
) -> (String, String, Option<String>, Option<String>) {
    let mut connection_string = String::new();
    let mut ssl_mode = "verify-full".to_string();
    let mut ssl_rootcert_path: Option<String> = None;
    let mut password: Option<String> = None;

    let str_params: Vec<&str> = pg_connection_string.split_whitespace().collect();
    for param in str_params {
        let param = param.split('=').collect::<Vec<&str>>();
        if let (Some(&name), Some(&value)) = (param.first(), param.get(1)) {
            match name {
                "sslmode" => {
                    ssl_mode = value.to_string();
                }
                "sslrootcert" => {
                    ssl_rootcert_path = Some(value.to_string());
                }
                "password" => {
                    password = Some(value.to_string());
                }
                _ => {
                    connection_string.push_str(format!("{name}={value} ").as_str());
                }
            }
        }
    }

    (connection_string, ssl_mode, ssl_rootcert_path, password)
}

/// The connection `params` configure, and whether to check first that its hosts resolve and
/// accept TCP: not when `hostaddr` pins the addresses, which are then all that is contacted.
fn config_from(params: &HashMap<String, SecretString>) -> Result<(Config, bool)> {
    let mut connection_string = match params
        .get("connection_string")
        .map(SecretBox::expose_secret)
    {
        Some(pg_connection_string) => parse_connection_string(pg_connection_string).0,
        None => {
            let mut connection_string = String::new();
            if let Some(pg_host) = params.get("host").map(SecretBox::expose_secret) {
                connection_string.push_str(format!("host={pg_host} ").as_str());
            }
            if let Some(pg_user) = params.get("user").map(SecretBox::expose_secret) {
                connection_string.push_str(format!("user={pg_user} ").as_str());
            }
            if let Some(pg_db) = params.get("db").map(SecretBox::expose_secret) {
                connection_string.push_str(format!("dbname={pg_db} ").as_str());
            }
            if let Some(pg_port) = params.get("port").map(SecretBox::expose_secret) {
                connection_string.push_str(format!("port={pg_port} ").as_str());
            }
            connection_string
        }
    };

    let mode = match ssl_mode(params)?.as_str() {
        "disable" => "disable",
        "prefer" => "prefer",
        // tokio_postgres supports only disable, require and prefer
        _ => "require",
    };

    // Password is never included in the connection string — it flows
    // through the PasswordProvider on each connection instead.
    connection_string.push_str(format!("sslmode={mode} ").as_str());
    let mut config = Config::from_str(connection_string.as_str()).context(ConnectionPoolSnafu)?;

    apply_optional_session_params(&mut config, params);
    apply_hostaddr(&mut config, params)?;

    let verify = config.get_hostaddrs().is_empty();
    Ok((config, verify))
}

/// Pins each host to the IP address at the same position in `hostaddr`: TCP goes to the
/// address, and TLS verifies the host name.
fn apply_hostaddr(config: &mut Config, params: &HashMap<String, SecretString>) -> Result<()> {
    if let Some(hostaddr) = params.get("hostaddr").map(SecretBox::expose_secret) {
        for addr in hostaddr.split(',') {
            config.hostaddr(addr.parse().ok().context(InvalidHostaddrSnafu)?);
        }
    }
    let hostaddrs = config.get_hostaddrs().len();
    ensure!(
        hostaddrs == 0 || hostaddrs == config.get_hosts().len(),
        InvalidHostaddrSnafu
    );
    Ok(())
}

/// The `sslmode` param, else the connection string's, else `verify-full`.
fn ssl_mode(params: &HashMap<String, SecretString>) -> Result<String> {
    if let Some(pg_sslmode) = params.get("sslmode").map(SecretBox::expose_secret) {
        let ssl_mode = pg_sslmode.to_lowercase();
        ensure!(
            matches!(
                ssl_mode.as_str(),
                "disable" | "require" | "prefer" | "verify-ca" | "verify-full"
            ),
            InvalidParameterSnafu {
                parameter_name: "sslmode".to_string(),
            }
        );
        return Ok(ssl_mode);
    }
    Ok(params
        .get("connection_string")
        .map(|s| parse_connection_string(s.expose_secret()).1)
        .unwrap_or_else(|| "verify-full".to_string()))
}

/// The root certificates trusted besides the platform's: the contents of the `sslrootcert`
/// file (the param's, else the connection string's), or `sslrootcert_pem` itself.
async fn root_certs(params: &HashMap<String, SecretString>) -> Result<Option<Vec<u8>>> {
    let path = params
        .get("sslrootcert")
        .map(|path| path.expose_secret().to_string())
        .or_else(|| {
            params
                .get("connection_string")
                .and_then(|s| parse_connection_string(s.expose_secret()).2)
        });
    match (path, params.get("sslrootcert_pem")) {
        (Some(_), Some(_)) => ConflictingRootCertsSnafu.fail(),
        (Some(path), None) => {
            ensure!(
                std::path::Path::new(&path).exists(),
                InvalidRootCertPathSnafu { path: &path }
            );
            Ok(Some(
                tokio::fs::read(path).await.context(FailedToReadCertSnafu)?,
            ))
        }
        (None, Some(pem)) => Ok(Some(pem.expose_secret().as_bytes().to_vec())),
        (None, None) => Ok(None),
    }
}

/// The password in the connection string, or else the `pass` param when there is none.
fn static_password(params: &HashMap<String, SecretString>) -> Option<SecretString> {
    match params.get("connection_string") {
        Some(s) => parse_connection_string(s.expose_secret())
            .3
            .map(SecretString::from),
        None => params.get("pass").cloned(),
    }
}

/// Apply the optional string session params libpq forwards verbatim to the
/// backend — `application_name` and `options` — onto a parsed [`Config`]. Split
/// out of `new_inner` so it is unit-testable without a live server.
///
/// `options` is the libpq `options` connection parameter: arbitrary command-line
/// switches sent at connection time (e.g. `-c statement_timeout=5000`), applied via
/// [`Config::options`]. It lets a caller pin session-level GUCs
/// (`statement_timeout`, `default_transaction_read_only`, …) that `Config` has no
/// dedicated setter for.
fn apply_optional_session_params(config: &mut Config, params: &HashMap<String, SecretString>) {
    if let Some(application_name) = params.get("application_name").map(SecretBox::expose_secret) {
        config.application_name(application_name);
    }
    if let Some(options) = params.get("options").map(SecretBox::expose_secret) {
        config.options(options);
    }
}

fn get_join_context(config: &Config) -> JoinPushDown {
    let mut join_push_context_str = String::new();
    for host in config.get_hosts() {
        join_push_context_str.push_str(&format!("host={host:?},"));
    }
    if !config.get_ports().is_empty() {
        join_push_context_str.push_str(&format!("port={port},", port = config.get_ports()[0]));
    }
    if let Some(dbname) = config.get_dbname() {
        join_push_context_str.push_str(&format!("db={dbname},"));
    }
    if let Some(user) = config.get_user() {
        join_push_context_str.push_str(&format!("user={user},"));
    }

    JoinPushDown::AllowedFor(join_push_context_str)
}

/// Classifies a connection error, returning a more specific error variant for
/// authentication failures.
fn classify_connection_error(err: tokio_postgres::Error) -> Error {
    if let Some(code) = err.code() {
        if *code == tokio_postgres::error::SqlState::INVALID_PASSWORD {
            return Error::InvalidUsernameOrPassword { source: err };
        }
    }
    Error::PostgresConnectionError { source: err }
}

async fn test_connection(config: &Config, connector: tls::Connector) -> Result<()> {
    config
        .connect(connector)
        .await
        .map(|_| ())
        .map_err(classify_connection_error)
}

fn map_pool_build_error(e: ConnectionManagerError) -> Error {
    match e {
        ConnectionManagerError::Postgres(e) => Error::ConnectionPoolError { source: e },
        ConnectionManagerError::PasswordProvider(e) => Error::PasswordProviderError { source: e },
    }
}

fn map_pool_run_error(e: bb8::RunError<ConnectionManagerError>) -> Error {
    match e {
        bb8::RunError::User(ConnectionManagerError::Postgres(e)) => Error::ConnectionPoolRunError {
            source: bb8::RunError::User(e),
        },
        bb8::RunError::User(ConnectionManagerError::PasswordProvider(e)) => {
            Error::PasswordProviderError { source: e }
        }
        bb8::RunError::TimedOut => Error::ConnectionPoolRunError {
            source: bb8::RunError::TimedOut,
        },
    }
}

async fn verify_postgres_config(config: &Config) -> Result<()> {
    for host in config.get_hosts() {
        for port in config.get_ports() {
            if let Host::Tcp(host) = host {
                verify_ns_lookup_and_tcp_connect(host, *port)
                    .await
                    .context(InvalidHostOrPortSnafu { host, port: *port })?;
            }
        }
    }

    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct PostgresErrorSink {}

impl PostgresErrorSink {
    pub fn new() -> Self {
        PostgresErrorSink {}
    }
}

impl<E> ErrorSink<E> for PostgresErrorSink
where
    E: std::fmt::Debug,
    E: std::fmt::Display,
{
    fn sink(&self, error: E) {
        tracing::debug!("Postgres Pool Error: {}", error);
    }

    fn boxed_clone(&self) -> Box<dyn ErrorSink<E>> {
        Box::new(*self)
    }
}

#[async_trait]
impl
    DbConnectionPool<bb8::PooledConnection<'static, ConnectionManager>, &'static (dyn ToSql + Sync)>
    for PostgresConnectionPool
{
    async fn connect(
        &self,
    ) -> std::result::Result<
        Box<
            dyn DbConnection<
                bb8::PooledConnection<'static, ConnectionManager>,
                &'static (dyn ToSql + Sync),
            >,
        >,
        Box<dyn std::error::Error + Send + Sync>,
    > {
        let pool = Arc::clone(&self.pool);
        let conn = if let Some(handle) = &self.io_handle {
            handle
                .spawn(async move { pool.get_owned().await.map_err(map_pool_run_error) })
                .await
                .context(IoRuntimeSnafu)??
        } else {
            let get_conn = async || pool.get_owned().await.map_err(map_pool_run_error);
            run_async_with_tokio(get_conn).await?
        };
        Ok(Box::new(
            PostgresConnection::new(conn)
                .with_unsupported_type_action(self.unsupported_type_action)
                .with_cancel_tls(self.tls.clone()),
        ))
    }

    fn join_push_down(&self) -> JoinPushDown {
        self.join_push_down.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use secrecy::ExposeSecret;
    use std::net::{IpAddr, Ipv6Addr};

    #[tokio::test]
    async fn static_password_provider_returns_password() {
        let provider = StaticPasswordProvider::new(SecretString::from("hunter2".to_string()));
        let password = provider.get_password().await.unwrap();
        assert_eq!(password.expose_secret(), "hunter2");
    }

    #[test]
    fn connection_manager_error_display() {
        let err = ConnectionManagerError::PasswordProvider("token expired".into());
        assert_eq!(err.to_string(), "password provider error: token expired");
    }

    #[test]
    fn connection_manager_error_implements_std_error() {
        let err: Box<dyn std::error::Error> =
            Box::new(ConnectionManagerError::PasswordProvider("fail".into()));
        assert!(err.source().is_some());
    }

    #[test]
    fn parse_connection_string_extracts_password() {
        let (conn_str, ssl_mode, cert_path, password) = parse_connection_string(
            "host=localhost user=postgres password=secret dbname=mydb sslmode=disable",
        );
        assert_eq!(conn_str.trim(), "host=localhost user=postgres dbname=mydb");
        assert_eq!(ssl_mode, "disable");
        assert!(cert_path.is_none());
        assert_eq!(password.as_deref(), Some("secret"));
    }

    #[test]
    fn parse_connection_string_without_password() {
        let (conn_str, _ssl_mode, _cert_path, password) =
            parse_connection_string("host=localhost user=postgres dbname=mydb");
        assert_eq!(conn_str.trim(), "host=localhost user=postgres dbname=mydb");
        assert!(password.is_none());
    }

    #[test]
    fn options_param_reaches_the_config() {
        let mut params: HashMap<String, SecretString> = HashMap::new();
        params.insert(
            "options".to_string(),
            SecretString::from("-c statement_timeout=5000".to_string()),
        );
        params.insert(
            "application_name".to_string(),
            SecretString::from("semvia".to_string()),
        );

        let mut config = Config::new();
        apply_optional_session_params(&mut config, &params);

        assert_eq!(config.get_options(), Some("-c statement_timeout=5000"));
        assert_eq!(config.get_application_name(), Some("semvia"));
    }

    fn params(entries: &[(&str, &str)]) -> HashMap<String, SecretString> {
        entries
            .iter()
            .map(|(name, value)| (name.to_string(), SecretString::from(value.to_string())))
            .collect()
    }

    const PINNED: &[(&str, &str)] = &[
        ("host", "customer.test"),
        ("hostaddr", "127.0.0.1"),
        ("port", "5432"),
        ("db", "gis"),
        ("user", "u"),
    ];

    fn params_without(name: &str) -> HashMap<String, SecretString> {
        let mut params = params(PINNED);
        params.remove(name);
        params
    }

    fn test_ca_pem() -> String {
        let mut params = rcgen::CertificateParams::new(Vec::<String>::new()).unwrap();
        params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
        rcgen::CertifiedIssuer::self_signed(params, rcgen::KeyPair::generate().unwrap())
            .unwrap()
            .pem()
    }

    #[test]
    fn hostaddr_pins_every_host_and_skips_the_lookup() {
        let (config, verify) = config_from(&params(&[
            ("host", "customer.test,customer.test"),
            ("hostaddr", "10.0.0.1,::1"),
        ]))
        .unwrap();
        assert_eq!(
            config.get_hostaddrs(),
            [
                IpAddr::from([10, 0, 0, 1]),
                IpAddr::from(Ipv6Addr::LOCALHOST)
            ]
        );
        assert!(!verify);

        let params = params(PINNED);
        let (config, verify) = config_from(&params).unwrap();
        assert_eq!(config.get_hostaddrs(), [IpAddr::from([127, 0, 0, 1])]);
        assert!(!verify, "a pinned pool never resolves or probes the host");
        let unpinned = config_from(&params_without("hostaddr")).unwrap();
        assert!(unpinned.1);
    }

    #[test]
    fn a_hostaddr_that_is_not_an_ip_literal_is_refused() {
        assert!(config_from(&params(&[("host", "h"), ("hostaddr", "db.internal")])).is_err());
        assert!(
            config_from(&params(&[("host", "h"), ("hostaddr", "10.0.0.1,10.0.0.2")])).is_err(),
            "one hostaddr per host"
        );
        assert!(
            config_from(&params(&[("hostaddr", "10.0.0.1")])).is_err(),
            "a hostaddr names the host TLS verifies"
        );
    }

    #[cfg(feature = "rustls")]
    #[test]
    fn in_memory_roots_are_parsed_and_bad_pem_is_refused() {
        assert!(tls::connector("verify-full", Some(test_ca_pem().as_bytes())).is_ok());
        assert!(tls::connector("verify-full", Some(b"not pem")).is_err());
    }

    #[tokio::test]
    async fn in_memory_roots_are_read_from_the_params() {
        let pem = test_ca_pem();
        let roots = root_certs(&params(&[("sslrootcert_pem", &pem)]))
            .await
            .unwrap();
        assert_eq!(roots.as_deref(), Some(pem.as_bytes()));
        assert!(
            root_certs(&params(&[
                ("sslrootcert_pem", &pem),
                ("sslrootcert", "root.crt")
            ]))
            .await
            .is_err(),
            "roots come from a file or from memory, not both"
        );
    }

    #[test]
    fn absent_options_param_leaves_the_config_untouched() {
        let params: HashMap<String, SecretString> = HashMap::new();

        let mut config = Config::new();
        apply_optional_session_params(&mut config, &params);

        assert_eq!(config.get_options(), None);
        assert_eq!(config.get_application_name(), None);
    }
}
