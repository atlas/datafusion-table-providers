//! Connecting over TLS in each `sslmode`, to a server presenting a certificate an authority
//! of the test's own issued for `localhost`. Runs against whichever TLS the build has.

use super::common;
use datafusion_table_providers::sql::db_connection_pool::postgrespool::PostgresConnectionPool;
use datafusion_table_providers::util::secrets::to_secret_map;
use rcgen::{
    BasicConstraints, CertificateParams, CertifiedIssuer, DnType, IsCa, KeyPair, KeyUsagePurpose,
};
use std::collections::HashMap;
use std::path::Path;

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

/// Turns TLS on, presenting a certificate `authority` issued for `localhost`.
async fn enable_tls(pool: &PostgresConnectionPool, authority: &CertifiedIssuer<'static, KeyPair>) {
    let key = KeyPair::generate().unwrap();
    let mut params = CertificateParams::new(vec!["localhost".to_string()]).unwrap();
    params
        .distinguished_name
        .push(DnType::CommonName, "localhost");
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

/// Host, `sslmode`, `sslrootcert`, and whether that connects and, if so, over TLS.
type Case<'a> = (&'a str, &'a str, Option<&'a Path>, Result<bool, ()>);

#[tokio::test]
async fn sslmodes_mean_what_they_mean_to_libpq() {
    let port = crate::get_random_port();
    let container = common::start_postgres_docker_container(port)
        .await
        .expect("Postgres container to start");

    let authority = authority();
    let admin = PostgresConnectionPool::new(to_secret_map(common::get_pg_params(port)))
        .await
        .expect("a plain connection");
    enable_tls(&admin, &authority).await;

    let dir = tempfile::tempdir().unwrap();
    let ours = dir.path().join("ours.pem");
    let theirs = dir.path().join("theirs.pem");
    std::fs::write(&ours, authority.pem()).unwrap();
    std::fs::write(&theirs, self::authority().pem()).unwrap();

    // The reload is asynchronous: wait until the server offers TLS.
    let mut offered = false;
    for _ in 0..50 {
        if connect(params(port, "localhost", "require", None)).await == Ok(true) {
            offered = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    assert!(offered, "the server should offer TLS once reloaded");

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
