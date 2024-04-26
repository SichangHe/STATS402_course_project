use pyo3::prelude::*;
use snork::{agents::TreeHeuristic, env::Direction, game::Game, search::max_n, util::argmax};

// Copied from `battlesnake_gym`.
const UP: isize = 0;
const RIGHT: isize = 1;
const DOWN: isize = 2;
const LEFT: isize = 3;

#[pyclass]
struct SnorkTreeAgent {
    heuristic: TreeHeuristic,
}

impl SnorkTreeAgent {
    pub fn step(&self, game: &Game, depth: usize) -> isize {
        let results = max_n(game, depth, &self.heuristic);
        argmax(results.iter().copied())
            .map(|d| match Direction::from(d as u8) {
                Direction::Up => UP,
                Direction::Right => RIGHT,
                Direction::Down => DOWN,
                Direction::Left => LEFT,
            })
            .unwrap_or(UP)
    }
}

#[pymethods]
impl SnorkTreeAgent {
    #[new]
    fn new() -> Self {
        let heuristic = TreeHeuristic::default();
        Self { heuristic }
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
    Ok(())
}
