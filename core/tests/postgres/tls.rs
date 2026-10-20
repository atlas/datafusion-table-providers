//! Connecting over TLS in each `sslmode`, to a server presenting a certificate an authority
//! of the test's own issued for `localhost`, and for a session's pinned host. Runs against
//! whichever TLS the build has.

use super::{abandon_a_statement, assert_backend_ends, common, session_pid};
use crate::docker::RunningContainer;
use datafusion_table_providers::sql::db_connection_pool::postgrespool::{
    self, PostgresConnectionPool,
};
use datafusion_table_providers::util::secrets::to_secret_map;
use rcgen::{
    BasicConstraints, CertificateParams, CertifiedIssuer, DnType, IsCa, KeyPair, KeyUsagePurpose,
};
use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

/// A certificate authority. Named apart from what it issues: OpenSSL takes a certificate
/// whose subject is its issuer's to be self-signed.
fn authority() -> CertifiedIssuer<'static, KeyPair> {
    let mut params = CertificateParams::new(Vec::<String>::new()).unwrap();
    params
        .distinguished_name
        .push(DnType::CommonName, "test authority");
    params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    params.key_usages = vec![KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign];
    CertifiedIssuer::self_signed(params, KeyPair::generate().unwrap()).unwrap()
}

/// Writes `pem` to `path` on the server, as the server's own user and readable by it alone,
/// as a private key must be: the server's own umask may let others read what `COPY` writes,
/// so it writes through a shell with a stricter one. One row per line.
async fn write_server_file(pool: &PostgresConnectionPool, path: &str, pem: &str) {
    let lines = pem
        .lines()
        .map(|line| format!("'{line}'"))
        .collect::<Vec<_>>()
        .join(",");
    pool.connect_direct()
        .await
        .unwrap()
        .conn
        .batch_execute(&format!(
            "COPY (SELECT unnest(ARRAY[{lines}])) TO PROGRAM 'umask 077 && cat > {path}'"
        ))
        .await
        .unwrap();
}

/// Turns TLS on, presenting a certificate `authority` issued for `names`.
async fn enable_tls(
    pool: &PostgresConnectionPool,
    authority: &CertifiedIssuer<'static, KeyPair>,
    names: &[&str],
) {
    let key = KeyPair::generate().unwrap();
    let mut params = CertificateParams::new(
        names
            .iter()
            .map(|name| name.to_string())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    params.distinguished_name.push(DnType::CommonName, names[0]);
    let cert = params.signed_by(&key, authority).unwrap();

    let conn = pool.connect_direct().await.unwrap();
    let dir: String = conn
        .conn
        .query_one("SELECT current_setting('data_directory')", &[])
        .await
        .unwrap()
        .get(0);
    let (cert_path, key_path) = (format!("{dir}/test.crt"), format!("{dir}/test.key"));
    write_server_file(pool, &cert_path, &cert.pem()).await;
    write_server_file(pool, &key_path, &key.serialize_pem()).await;

    // One at a time: `ALTER SYSTEM` refuses to run inside a transaction block, which a
    // multi-statement query implicitly is.
    for statement in [
        format!("ALTER SYSTEM SET ssl_cert_file = '{cert_path}'"),
        format!("ALTER SYSTEM SET ssl_key_file = '{key_path}'"),
        "ALTER SYSTEM SET ssl = on".to_string(),
        "SELECT pg_reload_conf()".to_string(),
    ] {
        conn.conn.batch_execute(&statement).await.unwrap();
    }
}

fn params(
    port: usize,
    host: &str,
    sslmode: &str,
    sslrootcert: Option<&Path>,
) -> HashMap<String, String> {
    let mut params = common::get_pg_params(port);
    params.insert("pg_host".to_string(), host.to_string());
    params.insert("pg_sslmode".to_string(), sslmode.to_string());
    if let Some(path) = sslrootcert {
        params.insert("pg_sslrootcert".to_string(), path.display().to_string());
    }
    params
}

/// [`params`], with the host pinned to the server's address.
fn pinned_params(
    port: usize,
    host: &str,
    sslmode: &str,
    sslrootcert: Option<&Path>,
) -> HashMap<String, String> {
    let mut params = params(port, host, sslmode, sslrootcert);
    params.insert("pg_hostaddr".to_string(), "127.0.0.1".to_string());
    params
}

/// Whether the pool connected, and if so whether over TLS.
async fn connect(params: HashMap<String, String>) -> Result<bool, String> {
    let pool = PostgresConnectionPool::new(to_secret_map(params))
        .await
        .map_err(|e| e.to_string())?;
    let conn = pool.connect_direct().await.map_err(|e| e.to_string())?;
    let row = conn
        .conn
        .query_one(
            "SELECT ssl FROM pg_stat_ssl WHERE pid = pg_backend_pid()",
            &[],
        )
        .await
        .map_err(|e| e.to_string())?;
    Ok(row.get(0))
}

/// Whether a session connected, and if so whether over TLS.
async fn connect_session(params: HashMap<String, String>) -> Result<bool, String> {
    let session =
        postgrespool::connect_session(to_secret_map(params), None, Duration::from_secs(30))
            .await
            .map_err(|e| e.to_string())?;
    let row = session
        .client
        .query_one(
            "SELECT ssl FROM pg_stat_ssl WHERE pid = pg_backend_pid()",
            &[],
        )
        .await
        .map_err(|e| e.to_string())?;
    Ok(row.get(0))
}

/// A server presenting a certificate `authority` issued for `names`, once it offers TLS.
async fn tls_server(
    port: usize,
    authority: &CertifiedIssuer<'static, KeyPair>,
    names: &[&str],
) -> RunningContainer {
    let container = common::start_postgres_docker_container(port)
        .await
        .expect("Postgres container to start");
    let admin = PostgresConnectionPool::new(to_secret_map(common::get_pg_params(port)))
        .await
        .expect("a plain connection");
    enable_tls(&admin, authority, names).await;

    // The reload is asynchronous: wait until the server offers TLS.
    for _ in 0..50 {
        if connect(params(port, "localhost", "require", None)).await == Ok(true) {
            return container;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("the server should offer TLS once reloaded");
}

/// Host, `sslmode`, `sslrootcert`, and whether that connects and, if so, over TLS.
type Case<'a> = (&'a str, &'a str, Option<&'a Path>, Result<bool, ()>);

#[tokio::test]
async fn sslmodes_mean_what_they_mean_to_libpq() {
    let port = crate::get_random_port();
    let authority = authority();
    let container = tls_server(port, &authority, &["localhost"]).await;

    let dir = tempfile::tempdir().unwrap();
    let ours = dir.path().join("ours.pem");
    let theirs = dir.path().join("theirs.pem");
    std::fs::write(&ours, authority.pem()).unwrap();
    std::fs::write(&theirs, self::authority().pem()).unwrap();

    let cases: [Case; 8] = [
        ("localhost", "disable", None, Ok(false)),
        ("localhost", "prefer", None, Ok(true)),
        // Encrypted, but nothing about the server is verified.
        ("localhost", "require", None, Ok(true)),
        ("localhost", "verify-full", Some(&ours), Ok(true)),
        // No root trusts our authority.
        ("localhost", "verify-full", None, Err(())),
        // The certificate is for `localhost`, not `127.0.0.1`.
        ("127.0.0.1", "verify-full", Some(&ours), Err(())),
        ("127.0.0.1", "verify-ca", Some(&ours), Ok(true)),
        ("localhost", "verify-ca", Some(&theirs), Err(())),
    ];
    for (host, sslmode, root, expected) in cases {
        let outcome = connect(params(port, host, sslmode, root)).await;
        assert_eq!(
            outcome.as_ref().map(|ssl| *ssl).map_err(|_| ()),
            expected,
            "host={host} sslmode={sslmode} sslrootcert={}: {outcome:?}",
            root.map_or("none".into(), |p| p.display().to_string()),
        );
    }

    container
        .remove()
        .await
        .expect("to stop postgres container");
}

/// A session reaches the server at its pinned `hostaddr` and verifies the server's
/// certificate against `host`, never the address. Dropped mid-statement, its cancel request
/// takes the same path and stops the statement.
#[tokio::test]
async fn a_session_connects_to_its_pinned_address_and_verifies_its_host() {
    let port = crate::get_random_port();
    let authority = authority();
    // `.invalid` resolves nowhere, so only the pinned address reaches the server.
    let container = tls_server(port, &authority, &["localhost", "customer.invalid"]).await;

    let dir = tempfile::tempdir().unwrap();
    let ours = dir.path().join("ours.pem");
    std::fs::write(&ours, authority.pem()).unwrap();

    let cases: [Case; 3] = [
        ("customer.invalid", "verify-full", Some(&ours), Ok(true)),
        // The certificate is not issued for this name, wherever it is reached.
        ("elsewhere.invalid", "verify-full", Some(&ours), Err(())),
        // No root trusts our authority.
        ("customer.invalid", "verify-full", None, Err(())),
    ];
    for (host, sslmode, root, expected) in cases {
        let outcome = connect_session(pinned_params(port, host, sslmode, root)).await;
        assert_eq!(
            outcome.as_ref().map(|ssl| *ssl).map_err(|_| ()),
            expected,
            "host={host} sslmode={sslmode} sslrootcert={}: {outcome:?}",
            root.map_or("none".into(), |p| p.display().to_string()),
        );
    }

    let admin = PostgresConnectionPool::new(to_secret_map(common::get_pg_params(port)))
        .await
        .expect("a plain connection")
        .connect_direct()
        .await
        .expect("Connection should be established");
    let session = postgrespool::connect_session(
        to_secret_map(pinned_params(
            port,
            "customer.invalid",
            "verify-full",
            Some(ours.as_path()),
        )),
        None,
        Duration::from_secs(30),
    )
    .await
    .expect("a pinned session should connect over TLS");
    let pid = session_pid(&session).await;
    abandon_a_statement(&session, &admin).await;
    drop(session);
    assert_backend_ends(&admin, pid).await;

    container
        .remove()
        .await
        .expect("to stop postgres container");
}
