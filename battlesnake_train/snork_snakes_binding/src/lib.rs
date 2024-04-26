use std::{sync::Arc, time::Duration};

use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};
use snork::{
    agents::TreeHeuristic,
    env::Direction,
    game::Game,
    search::{async_max_n, Heuristic},
    util::argmax,
};
use tokio::{runtime::Runtime, time::timeout};

// Copied from `battlesnake_gym`.
const N_SNAKES: usize = 4;
const UP: isize = 0;
const RIGHT: isize = 1;
const DOWN: isize = 2;
const LEFT: isize = 3;

const FIVE_SECONDS: Duration = Duration::from_secs(5);

#[pyclass]
struct SnorkTreeAgent {
    heuristic: Arc<dyn Heuristic>,
    runtime: Runtime,
}

#[pymethods]
impl SnorkTreeAgent {
    #[new]
    fn new() -> Self {
        let heuristic = Arc::new(TreeHeuristic::default());
        let runtime = Runtime::new().unwrap();
        Self { heuristic, runtime }
    }

    fn step(
        &self,
        game: Bound<PyBytes>,
        agents: [bool; N_SNAKES],
        depth: usize,
    ) -> PyResult<[isize; N_SNAKES]> {
        let mut game: Game = bincode::deserialize(game.as_bytes()).map_err(|why| {
            PyValueError::new_err(format!(
                "`snork_snakes_binding::SnorkTreeAgent::step` Error deserializing game: `{why}`"
            ))
        })?;

        let mut results = [UP; N_SNAKES];
        for (i, &enabled) in agents.iter().enumerate() {
            if enabled {
                // Make agent `i` the first, and perform the heuristic.
                game.snakes.swap(0, i);
                let future = step_timeout(self.heuristic.clone(), &game, depth);
                results[i] = self.runtime.block_on(future);
                game.snakes.swap(0, i);
            }
        }
        Ok(results)
    }
}

/// Work around the bug in `snork` that the first step takes forever.
async fn step_timeout(heuristic: Arc<dyn Heuristic>, game: &Game, depth: usize) -> isize {
    timeout(FIVE_SECONDS, step(heuristic, game, depth))
        .await
        .unwrap_or_else(|_| {
            game.valid_moves(0)
                .next()
                .map(direction_to_int)
                .unwrap_or(UP)
        })
}

async fn step(heuristic: Arc<dyn Heuristic>, game: &Game, depth: usize) -> isize {
    let results = async_max_n(game, depth, heuristic).await;
    argmax(results.iter().copied())
        .map(|d| direction_to_int(Direction::from(d as u8)))
        .unwrap_or(UP)
}

fn direction_to_int(direction: Direction) -> isize {
    match direction {
        Direction::Up => UP,
        Direction::Right => RIGHT,
        Direction::Down => DOWN,
        Direction::Left => LEFT,
    }
}

/// Prints a message.
#[pyfunction]
fn hello() -> PyResult<String> {
    Ok("Hello from snork-snakes-binding!".into())
}

/// A Python module implemented in Rust.
#[pymodule]
fn _lowlevel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_class::<SnorkTreeAgent>()?;
    Ok(())
}
