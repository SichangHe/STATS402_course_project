use axum::{
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};

use crate::{info::Info, logic::make_move};

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
    let move_to_take = do_handle_move(game_state).unwrap_or_else(|why| {
        warn!("Failed to make move, moving up: {:?}", why);
        Move::Up
    });
    Json(move_to_take)
}

fn do_handle_move(game_state: Game) -> Result<Move, Box<dyn Error>> {
    let snake_ids = build_snake_id_map(&game_state);
    let compact_game = StandardCellBoard4Snakes11x11::convert_from_game(game_state, &snake_ids)?;

    Ok(make_move(&compact_game))
}

#[instrument]
async fn handle_end(Json(game_state): Json<Game>) -> StatusCode {
    warn!("End.");
    StatusCode::OK
}
