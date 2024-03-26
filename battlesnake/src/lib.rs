use anyhow::Result;
use serde::Serialize;
use tracing::*;
use tracing_subscriber::EnvFilter;

pub mod info;
pub mod serve;

#[tokio::main]
pub async fn run(port: &str) {
    serve::serve(port).await.unwrap();
}

pub fn init_info_tracing_from_env() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(Level::INFO.into())
                .from_env_lossy(),
        )
        .init();
}
