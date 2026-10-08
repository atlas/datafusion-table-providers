//! The TLS a connection pool connects with: rustls, native-tls, or none, by cargo feature.
//! With both features, rustls is used.
//!
//! `sslmode` means what it means to libpq:
//!
//! - `disable`, `prefer`, `require`: encrypt (not at all, if the server will, always) but
//!   verify nothing about the server, so an active attacker can impersonate it.
//! - `verify-ca`: the server's certificate chains to a trusted root, whatever name it is
//!   issued for.
//! - `verify-full`: that, and it is issued for the host connected to.
//!
//! The trusted roots are the platform's, and those in the `sslrootcert` file or in
//! `sslrootcert_pem`.

pub(crate) use backend::{connector, Connector};

#[cfg(feature = "rustls")]
mod backend {
    use crate::pool::{BoxError, FailedToBuildTlsConnectorSnafu, FailedToLoadCertSnafu, Result};
    use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
    use rustls::client::WebPkiServerVerifier;
    use rustls::crypto::{self, CryptoProvider};
    use rustls::pki_types::pem::PemObject;
    use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
    use rustls::{
        CertificateError, ClientConfig, DigitallySignedStruct, RootCertStore, SignatureScheme,
    };
    use snafu::prelude::*;
    use std::sync::Arc;

    pub(crate) type Connector = tokio_postgres_rustls::MakeRustlsConnect;

    /// The connector for `ssl_mode`, trusting the certificates in `root_certs` (the contents
    /// of an `sslrootcert` file, or `sslrootcert_pem`) besides the platform's.
    pub(crate) fn connector(ssl_mode: &str, root_certs: Option<&[u8]>) -> Result<Connector> {
        // Named rather than taken from the process default, which is ambiguous — and
        // panics — when a build links more than one provider.
        let provider = Arc::new(crypto::ring::default_provider());

        let verifier: Arc<dyn ServerCertVerifier> = match ssl_mode {
            "verify-full" => verified_chain(&provider, root_certs)?,
            "verify-ca" => Arc::new(AnyName(verified_chain(&provider, root_certs)?)),
            _ => Arc::new(AnyCertificate(provider.clone())),
        };

        let config = ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .map_err(BoxError::from)
            .context(FailedToBuildTlsConnectorSnafu)?
            .dangerous()
            .with_custom_certificate_verifier(verifier)
            .with_no_client_auth();

        Ok(Connector::new(config))
    }

    /// Verifies the server's certificate chains to a trusted root and names the host.
    fn verified_chain(
        provider: &Arc<CryptoProvider>,
        root_certs: Option<&[u8]>,
    ) -> Result<Arc<WebPkiServerVerifier>> {
        let mut roots = RootCertStore::empty();

        let native = rustls_native_certs::load_native_certs();
        for error in &native.errors {
            tracing::debug!("could not load a platform root certificate: {error}");
        }
        roots.add_parsable_certificates(native.certs);

        if let Some(buf) = root_certs {
            for cert in parse_certs(buf)? {
                roots
                    .add(cert)
                    .map_err(BoxError::from)
                    .context(FailedToLoadCertSnafu)?;
            }
        }

        WebPkiServerVerifier::builder_with_provider(Arc::new(roots), provider.clone())
            .build()
            .map_err(BoxError::from)
            .context(FailedToBuildTlsConnectorSnafu)
    }

    /// The certificates in an `sslrootcert` file: any number in PEM, or one in DER.
    pub(super) fn parse_certs(buf: &[u8]) -> Result<Vec<CertificateDer<'static>>> {
        let pem = CertificateDer::pem_slice_iter(buf)
            .collect::<Result<Vec<_>, _>>()
            .unwrap_or_default();
        if !pem.is_empty() {
            return Ok(pem);
        }
        // Not PEM, so DER, whose validity only adding it to the store establishes.
        if buf.is_empty() {
            return Err(crate::pool::Error::FailedToLoadCertError {
                source: BoxError::from("the root certificate file is empty"),
            });
        }
        Ok(vec![CertificateDer::from(buf.to_vec())])
    }

    /// `verify-ca`: the chain is verified in full, then a certificate issued for another
    /// name is accepted.
    ///
    /// Sound because `WebPkiServerVerifier` checks the chain, and revocation, before the
    /// name: a name mismatch is only ever reported for a chain that verified.
    #[derive(Debug)]
    pub(super) struct AnyName(pub(super) Arc<WebPkiServerVerifier>);

    impl ServerCertVerifier for AnyName {
        fn verify_server_cert(
            &self,
            end_entity: &CertificateDer<'_>,
            intermediates: &[CertificateDer<'_>],
            server_name: &ServerName<'_>,
            ocsp_response: &[u8],
            now: UnixTime,
        ) -> Result<ServerCertVerified, rustls::Error> {
            match self.0.verify_server_cert(
                end_entity,
                intermediates,
                server_name,
                ocsp_response,
                now,
            ) {
                Err(rustls::Error::InvalidCertificate(
                    CertificateError::NotValidForName
                    | CertificateError::NotValidForNameContext { .. },
                )) => Ok(ServerCertVerified::assertion()),
                verified => verified,
            }
        }

        fn verify_tls12_signature(
            &self,
            message: &[u8],
            cert: &CertificateDer<'_>,
            dss: &DigitallySignedStruct,
        ) -> Result<HandshakeSignatureValid, rustls::Error> {
            self.0.verify_tls12_signature(message, cert, dss)
        }

        fn verify_tls13_signature(
            &self,
            message: &[u8],
            cert: &CertificateDer<'_>,
            dss: &DigitallySignedStruct,
        ) -> Result<HandshakeSignatureValid, rustls::Error> {
            self.0.verify_tls13_signature(message, cert, dss)
        }

        fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
            self.0.supported_verify_schemes()
        }
    }

    /// `require` and below: any certificate is accepted. The handshake is still checked to
    /// be signed by the key the certificate presents, so the session is encrypted to
    /// whoever holds it — just not verified to be the server.
    #[derive(Debug)]
    pub(super) struct AnyCertificate(pub(super) Arc<CryptoProvider>);

    impl ServerCertVerifier for AnyCertificate {
        fn verify_server_cert(
            &self,
            _end_entity: &CertificateDer<'_>,
            _intermediates: &[CertificateDer<'_>],
            _server_name: &ServerName<'_>,
            _ocsp_response: &[u8],
            _now: UnixTime,
        ) -> Result<ServerCertVerified, rustls::Error> {
            Ok(ServerCertVerified::assertion())
        }

        fn verify_tls12_signature(
            &self,
            message: &[u8],
            cert: &CertificateDer<'_>,
            dss: &DigitallySignedStruct,
        ) -> Result<HandshakeSignatureValid, rustls::Error> {
            crypto::verify_tls12_signature(
                message,
                cert,
                dss,
                &self.0.signature_verification_algorithms,
            )
        }

        fn verify_tls13_signature(
            &self,
            message: &[u8],
            cert: &CertificateDer<'_>,
            dss: &DigitallySignedStruct,
        ) -> Result<HandshakeSignatureValid, rustls::Error> {
            crypto::verify_tls13_signature(
                message,
                cert,
                dss,
                &self.0.signature_verification_algorithms,
            )
        }

        fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
            self.0.signature_verification_algorithms.supported_schemes()
        }
    }
}

#[cfg(all(feature = "native-tls", not(feature = "rustls")))]
mod backend {
    use crate::pool::{BoxError, FailedToBuildTlsConnectorSnafu, FailedToLoadCertSnafu, Result};
    use native_tls::{Certificate, TlsConnector};
    use snafu::prelude::*;

    pub(crate) type Connector = postgres_native_tls::MakeTlsConnector;

    /// The connector for `ssl_mode`, trusting the certificates in `root_certs` (the contents
    /// of an `sslrootcert` file, or `sslrootcert_pem`) besides the platform's.
    pub(crate) fn connector(ssl_mode: &str, root_certs: Option<&[u8]>) -> Result<Connector> {
        let mut builder = TlsConnector::builder();

        if ssl_mode != "disable" {
            if let Some(buf) = root_certs {
                for cert in parse_certs(buf)? {
                    builder.add_root_certificate(cert);
                }
            }
            builder
                .danger_accept_invalid_hostnames(ssl_mode != "verify-full")
                .danger_accept_invalid_certs(ssl_mode != "verify-full" && ssl_mode != "verify-ca");
        }

        let connector = builder
            .build()
            .map_err(BoxError::from)
            .context(FailedToBuildTlsConnectorSnafu)?;
        Ok(Connector::new(connector))
    }

    /// The certificates in an `sslrootcert` file: one in DER, or any number in PEM.
    fn parse_certs(buf: &[u8]) -> Result<Vec<Certificate>> {
        Certificate::from_der(buf)
            .map(|x| vec![x])
            .or_else(|_| {
                pem::parse_many(buf)
                    .unwrap_or_default()
                    .iter()
                    .map(pem::encode)
                    .map(|s| Certificate::from_pem(s.as_bytes()))
                    .collect()
            })
            .map_err(BoxError::from)
            .context(FailedToLoadCertSnafu)
    }
}

#[cfg(not(any(feature = "native-tls", feature = "rustls")))]
mod backend {
    use crate::pool::{Result, TlsNotCompiledSnafu};
    use snafu::prelude::*;

    pub(crate) type Connector = tokio_postgres::NoTls;

    /// Plain text only: `prefer` connects unencrypted, and anything that insists on TLS is
    /// refused rather than quietly downgraded.
    pub(crate) fn connector(ssl_mode: &str, _root_certs: Option<&[u8]>) -> Result<Connector> {
        ensure!(
            matches!(ssl_mode, "disable" | "prefer"),
            TlsNotCompiledSnafu { ssl_mode }
        );
        Ok(tokio_postgres::NoTls)
    }
}

#[cfg(all(test, feature = "rustls"))]
mod tests {
    use super::backend::{connector, parse_certs, AnyCertificate, AnyName};
    use rcgen::{BasicConstraints, CertificateParams, CertifiedIssuer, IsCa, KeyPair};
    use rustls::client::danger::ServerCertVerifier;
    use rustls::client::WebPkiServerVerifier;
    use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
    use rustls::{CertificateError, RootCertStore};
    use std::sync::Arc;

    fn authority() -> CertifiedIssuer<'static, KeyPair> {
        let mut params = CertificateParams::new(Vec::<String>::new()).unwrap();
        params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        CertifiedIssuer::self_signed(params, KeyPair::generate().unwrap()).unwrap()
    }

    fn server(
        authority: &CertifiedIssuer<'static, KeyPair>,
        name: &str,
    ) -> CertificateDer<'static> {
        let params = CertificateParams::new(vec![name.to_string()]).unwrap();
        params
            .signed_by(&KeyPair::generate().unwrap(), authority)
            .unwrap()
            .der()
            .clone()
    }

    fn chain_verifier(authority: &CertifiedIssuer<'static, KeyPair>) -> Arc<WebPkiServerVerifier> {
        let mut roots = RootCertStore::empty();
        roots.add(authority.der().clone()).unwrap();
        WebPkiServerVerifier::builder_with_provider(
            Arc::new(roots),
            Arc::new(rustls::crypto::ring::default_provider()),
        )
        .build()
        .unwrap()
    }

    fn verify(
        verifier: &dyn ServerCertVerifier,
        cert: &CertificateDer<'_>,
        host: &str,
    ) -> Result<(), rustls::Error> {
        let name = ServerName::try_from(host.to_string()).unwrap();
        verifier
            .verify_server_cert(cert, &[], &name, &[], UnixTime::now())
            .map(|_| ())
    }

    #[test]
    fn verify_full_needs_the_chain_and_the_name() {
        let authority = authority();
        let verifier = chain_verifier(&authority);
        let cert = server(&authority, "localhost");

        assert!(verify(verifier.as_ref(), &cert, "localhost").is_ok());
        let mismatch = verify(verifier.as_ref(), &cert, "db.example.com");
        assert!(
            matches!(
                mismatch,
                Err(rustls::Error::InvalidCertificate(
                    CertificateError::NotValidForName
                        | CertificateError::NotValidForNameContext { .. }
                ))
            ),
            "{mismatch:?}"
        );
    }

    #[test]
    fn verify_ca_forgives_the_name_and_nothing_else() {
        let authority = authority();
        let verifier = AnyName(chain_verifier(&authority));

        assert!(verify(
            &verifier,
            &server(&authority, "localhost"),
            "db.example.com"
        )
        .is_ok());
        assert!(verify(&verifier, &server(&authority, "localhost"), "127.0.0.1").is_ok());
        assert!(
            verify(
                &verifier,
                &server(&self::authority(), "localhost"),
                "localhost"
            )
            .is_err(),
            "a certificate from another authority is refused"
        );
    }

    #[test]
    fn require_accepts_any_certificate() {
        let verifier = AnyCertificate(Arc::new(rustls::crypto::ring::default_provider()));
        assert!(verify(&verifier, &server(&authority(), "anything"), "localhost").is_ok());
    }

    #[test]
    fn a_root_file_is_pem_or_der() {
        let (one, two) = (authority(), authority());
        let pem = format!("{}{}", one.pem(), two.pem());
        assert_eq!(parse_certs(pem.as_bytes()).unwrap().len(), 2);
        assert_eq!(parse_certs(one.der()).unwrap(), vec![one.der().clone()]);
        assert!(parse_certs(b"").is_err());
    }

    #[test]
    fn a_root_file_that_is_no_certificate_is_refused() {
        let Err(err) = connector("verify-full", Some(b"not a certificate")) else {
            panic!("a root file that is no certificate should be refused");
        };
        assert!(
            err.to_string().contains("Certificate loading failed"),
            "{err}"
        );
    }

    #[test]
    fn modes_that_verify_nothing_need_no_roots() {
        for mode in ["disable", "prefer", "require"] {
            assert!(connector(mode, None).is_ok(), "{mode}");
        }
    }
}
