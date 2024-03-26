use std::error::Error;

use anyhow::Result;
use battlesnake_game_types::{
    compact_representation::StandardCellBoard4Snakes11x11,
    types::{build_snake_id_map, Move},
    wire_representation::Game,
};
use serde::Serialize;
use tracing::*;
use tracing_subscriber::EnvFilter;

pub mod info;
pub mod logic;
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
