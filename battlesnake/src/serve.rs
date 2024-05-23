use axum::{
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use snork_engine::game::Snake;

use crate::{info::Info, logic::respond_move, model::Model};

use super::*;

#[instrument]
pub async fn serve(port: &str) -> Result<()> {
    pyo3::prepare_freethreaded_python();
    let model = Arc::new(Model::try_new()?);
    let app = Router::new()
        .route("/", get(handle_index))
        .route("/start", post(handle_start))
        .route(
            "/move",
            post(move |game_request| handle_move(game_request, model.clone())),
        )
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

const LATENCY_MS: u64 = 80;

#[instrument(skip(game_request, model))]
async fn handle_move(
    Json(game_request): Json<GameRequest>,
    model: Arc<Model>,
) -> Json<MoveResponse> {
    info!(?game_request, "Move.");
    let mut game = Game::from_request(&game_request);
    fix_game(&mut game);
    let timeout = Duration::from_millis(game_request.game.timeout.saturating_sub(LATENCY_MS));
    let move_to_take = respond_move(&game, timeout, model)
        .await
        .unwrap_or_else(|why| {
            let direction = game.valid_moves(0).next().unwrap_or(Direction::Up);
            warn!("Failed to make move, moving `{direction}`: {why:?}");
            MoveResponse::new(direction)
        });
    Json(move_to_take)
}

fn fix_game(game: &mut Game) {
    if game.snakes.len() < 4 {
        (0..(4 - game.snakes.len())).for_each(|_| {
            let dead_snake = Snake::new(Default::default(), 0);
            game.snakes.push(dead_snake);
        });
    }
}

#[instrument]
async fn handle_end(Json(game_request): Json<GameRequest>) -> StatusCode {
    warn!("End.");
    StatusCode::OK
}
