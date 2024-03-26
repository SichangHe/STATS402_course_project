use axum::{
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use battlesnake_game_types::{types::Move, wire_representation::Game};

use self::info::Info;

use super::*;

#[instrument]
pub async fn serve(port: &str) -> Result<()> {
    let app = Router::new()
        .route("/", get(handle_index))
        .route("/start", post(handle_start))
        .route("/move", post(handle_move))
        .route("/end", post(handle_end));

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}")).await?;
    warn!("Listening.");
    axum::serve(listener, app).await?;
    Ok(())
}

#[instrument]
async fn handle_index() -> Json<Info> {
    info!("/");
    Json(Info::ME)
}

#[instrument]
async fn handle_start(Json(game_state): Json<Game>) -> StatusCode {
    warn!("Start.");
    StatusCode::OK
}

#[instrument]
async fn handle_move(Json(game_state): Json<Game>) -> Json<Move> {
    info!("Move.");
    // TODO: Implement logic.
    Json(Move::Up)
}

#[instrument]
async fn handle_end(Json(game_state): Json<Game>) -> StatusCode {
    warn!("End.");
    StatusCode::OK
}
