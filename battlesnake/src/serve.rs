use axum::{
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};

use crate::{info::Info, logic::respond_move};

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
async fn handle_start(Json(game_request): Json<GameRequest>) -> StatusCode {
    warn!("Start.");
    StatusCode::OK
}

const LATENCY_MS: u64 = 40;

#[instrument]
async fn handle_move(Json(game_request): Json<GameRequest>) -> Json<MoveResponse> {
    info!("Move.");
    let game = Game::from_request(&game_request);
    let timeout = Duration::from_millis(game_request.game.timeout.saturating_sub(LATENCY_MS));
    let move_to_take = respond_move(&game, timeout).await.unwrap_or_else(|why| {
        let direction = game.valid_moves(0).next().unwrap_or(Direction::Up);
        warn!("Failed to make move, moving `{direction}`: {why:?}");
        MoveResponse::new(direction)
    });
    Json(move_to_take)
}

#[instrument]
async fn handle_end(Json(game_request): Json<GameRequest>) -> StatusCode {
    warn!("End.");
    StatusCode::OK
}
