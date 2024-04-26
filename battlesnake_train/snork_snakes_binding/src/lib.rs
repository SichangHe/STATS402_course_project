use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};
use snork::{agents::TreeHeuristic, env::Direction, game::Game, search::max_n, util::argmax};

// Copied from `battlesnake_gym`.
const N_SNAKES: usize = 4;
const UP: isize = 0;
const RIGHT: isize = 1;
const DOWN: isize = 2;
const LEFT: isize = 3;

#[pyclass]
struct SnorkTreeAgent {
    heuristic: TreeHeuristic,
}

#[pymethods]
impl SnorkTreeAgent {
    #[new]
    fn new() -> Self {
        let heuristic = TreeHeuristic::default();
        Self { heuristic }
    }

    fn step(
        &self,
        game: Bound<PyBytes>,
        agents: [bool; N_SNAKES],
        depth: usize,
    ) -> PyResult<[isize; N_SNAKES]> {
        let game = bincode::deserialize(game.as_bytes()).map_err(|why| {
            PyValueError::new_err(format!(
                "`snork_snakes_binding::SnorkTreeAgent::step` Error deserializing game: `{why}`"
            ))
        })?;

        let mut results = [UP; N_SNAKES];
        for (i, &agent) in agents.iter().enumerate() {
            if agent {
                results[i] = step(&self.heuristic, &game, depth);
            }
        }
        Ok(results)
    }
}

pub fn step(heuristic: &TreeHeuristic, game: &Game, depth: usize) -> isize {
    let results = max_n(game, depth, heuristic);
    argmax(results.iter().copied())
        .map(|d| match Direction::from(d as u8) {
            Direction::Up => UP,
            Direction::Right => RIGHT,
            Direction::Down => DOWN,
            Direction::Left => LEFT,
        })
        .unwrap_or(UP)
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
