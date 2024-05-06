use std::{cmp::Ordering, f64::consts::LN_2, marker::PhantomData, sync::Arc, time::Duration};

use anyhow::{Context, Result};
use battlesnake_gym::{
    direction2snake_true_move, snake_true_move2direction, LOSE_REWARD, UP, WIN_REWARD,
};
use futures::prelude::*;
use itertools::Itertools;
use pyo3::prelude::*;
use serde::Serialize;
use shame::*;
use snork_engine::{
    env::{Direction, GameRequest, MoveResponse},
    game::{Game, Outcome},
};
use tinyvec::ArrayVec;
use tracing::*;
use tracing_subscriber::EnvFilter;

pub mod info;
pub mod logic;
mod model;
pub mod serve;

#[tokio::main]
pub async fn run(port: &str) {
    serve::serve(port).await.unwrap();
}

pub fn try_init_info_tracing_from_env() {
    _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(Level::INFO.into())
                .from_env_lossy(),
        )
        .try_init();
}
