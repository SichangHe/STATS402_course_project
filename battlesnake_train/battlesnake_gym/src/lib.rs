use std::{
    cmp::Ordering::{Equal, Greater, Less},
    ops::Range,
};

use itertools::Itertools;
use numpy::{ndarray::Array3, IntoPyArray, PyArray3};
use pyo3::{exceptions::PyValueError, import_exception, prelude::*, types::PyBytes};
use rand::{seq::IteratorRandom, Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use snork_engine::{
    env::{Direction, Vec2D},
    game::{Game, Snake},
    grid::CellT,
    simulate::init_game,
};
use tracing::*;
use tracing_subscriber::EnvFilter;

mod observations;

use observations::*;

const BOARD_SIZE: usize = 11;
const N_SNAKES: usize = 4;
pub const WIN_REWARD: f64 = 1.0;
pub const LOSE_REWARD: f64 = -1.0;
const SURVIVE_ROUND_REWARD: f64 = 0.002;
const EATING_REWARD_FACTOR: f64 = 0.000_001;

// Clockwise, opposite to `np.rot90`.
pub const UP: isize = 0;
pub const RIGHT: isize = 1;
pub const DOWN: isize = 2;
pub const LEFT: isize = 3;

const PADDED_SIZE: usize = BOARD_SIZE * 2 - 1;
const N_LAYERS: usize = 9;
const HEALTH_NORMALIZATION: f32 = 1.0 / 239.0;

type BoundArray3<'py> = Bound<'py, PyArray3<f32>>;

#[pyclass]
#[derive(Clone, Debug, Deserialize, Serialize)]
struct SnakeGame {
    pub game: Game,
    pub rng: ChaCha8Rng,
    pub food_count: u16,
    pub eating_reward_factor: f64,
}

#[pymethods]
impl SnakeGame {
    #[new]
    fn new() -> PyResult<Self> {
        let mut rng = ChaCha8Rng::from_entropy();
        let game = fresh_game(&mut rng);
        Ok(Self {
            game,
            rng,
            food_count: N_SNAKES as u16,
            eating_reward_factor: EATING_REWARD_FACTOR,
        })
    }

    fn reset(&mut self) -> PyResult<()> {
        self.game = fresh_game(&mut self.rng);
        self.food_count = N_SNAKES as u16;
        Ok(())
    }

    /// Returns (`rewards`, `terminations`).
    fn step(
        &mut self,
        actions: [isize; N_SNAKES],
    ) -> PyResult<([f64; N_SNAKES], [bool; N_SNAKES])> {
        let health_before: Vec<_> = self.game.snakes.iter().map(|s| s.health).collect();

        // Based on `snork/src/simulate.rs`.
        let mut moves = [Direction::Up; N_SNAKES];
        for (index, &relative_move) in actions.iter().enumerate() {
            moves[index] =
                snake_true_move2direction(snake_true_move(&self.game.snakes[index], relative_move));
        }
        self.game.step(&moves);

        // Based on `snork/src/game.rs`.
        let mut n_alive = 0usize;
        let mut rewards = [0.0; N_SNAKES];
        let mut terminations = [true; N_SNAKES];
        let mut survivor = 0;
        for (i, snake) in self.game.snakes.iter().enumerate() {
            if snake.alive() {
                // Stayed alive.
                n_alive += 1;
                survivor = i;
                rewards[i] = SURVIVE_ROUND_REWARD;
                terminations[i] = false;
            } else if health_before[i] > 0 {
                // Died this round.
                rewards[i] = LOSE_REWARD;
            } // Else: already dead last round.
        }
        match n_alive {
            0 => {}
            1 => {
                rewards[survivor] = WIN_REWARD;
                terminations[survivor] = true;
            }
            _ => {
                // Check if snakes have consumed food
                for (i, snake) in self.game.snakes.iter().enumerate() {
                    if snake.health == 100 {
                        self.food_count -= 1;
                        rewards[i] += eating_reward(self.eating_reward_factor, health_before[i]);
                    }
                }

                // Spawn food
                if self.food_count == 0 || self.rng.gen::<f64>() < FOOD_RATE {
                    if let Some(cell) = self
                        .game
                        .grid
                        .cells
                        .iter_mut()
                        .filter(|c| c.t == CellT::Free)
                        .choose(&mut self.rng)
                    {
                        cell.t = CellT::Food;
                        self.food_count += 1;
                    }
                }
            }
        }

        Ok((rewards, terminations))
    }

    /// Observation states and facings for each snake.
    fn states<'py>(&self, py: Python<'py>) -> PyResult<(Vec<BoundArray3<'py>>, Vec<isize>)> {
        let (raw_states, snake_facings) = states(&self.game);
        let states = raw_states
            .into_iter()
            .map(|state| state.into_pyarray_bound(py))
            .collect::<Vec<_>>();
        Ok((states, snake_facings))
    }

    fn render(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.game))
    }

    /// Given the true move of a snake, return the move relative to its facing.
    fn snake_true_move(&self, snake_index: usize, relative_move: isize) -> PyResult<isize> {
        Ok(snake_true_move(
            &self.game.snakes[snake_index],
            relative_move,
        ))
    }

    /// Given the move of a snake relative to its facing, return the true move.
    fn snake_relative_move(&self, snake_index: usize, true_move: isize) -> PyResult<isize> {
        Ok(snake_relative_move(
            &self.game.snakes[snake_index],
            true_move,
        ))
    }

    fn game_serialized<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        bincode::serialize(&self.game)
            .map_err(|why| {
                PyValueError::new_err(format!(
                    "Error serializing `battlesnake_gym::SnakeGame.game`: `{why}`"
                ))
            })
            .map(|bytes| PyBytes::new_bound(py, &bytes))
    }

    /// Eating reward factor $f_e$ in $R_e = f_e(101 - h)$ where
    /// $h$ is the health of a snake and $R_e$ is the eating reward.
    fn set_eating_reward_factor(&mut self, eating_reward_factor: f64) -> PyResult<()> {
        self.eating_reward_factor = eating_reward_factor;
        Ok(())
    }

    // Referencing <https://github.com/light-curve/light-curve-python/pull/145/files>.
    /// Used by pickle.load / pickle.loads
    fn __setstate__(&mut self, state: Bound<PyBytes>) -> PyResult<()> {
        *self = bincode::deserialize(state.as_bytes()).map_err(|why| {
            import_exception!(pickle, UnpicklingError);
            UnpicklingError::new_err(format!(
                "Error deserializing `battlesnake_gym::SnakeGame`: `{why}`"
            ))
        })?;
        Ok(())
    }

    /// Used by pickle.dump / pickle.dumps
    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let vec_bytes = bincode::serialize(&self).map_err(|why| {
            import_exception!(pickle, PicklingError);
            PicklingError::new_err(format!(
                "Error serializing  `battlesnake_gym::SnakeGame`: `{why}`"
            ))
        })?;
        Ok(PyBytes::new_bound(py, &vec_bytes))
    }

    /// Used by copy.copy
    fn __copy__(&self) -> Self {
        self.clone()
    }

    /// Used by copy.deepcopy
    fn __deepcopy__(&self, _memo: Bound<PyAny>) -> Self {
        self.clone()
    }
}

fn snake_facing(snake: &Snake) -> Option<isize> {
    let mut head_to_tail = snake.body.iter().rev();
    match (head_to_tail.next(), head_to_tail.next()) {
        (Some(head), Some(neck)) => match (head.x.cmp(&neck.x), head.y.cmp(&neck.y)) {
            (Less, Equal) => Some(LEFT),
            (Equal, Less) => Some(DOWN),
            (Equal, Equal) => None,
            (Equal, Greater) => Some(UP),
            (Greater, Equal) => Some(RIGHT),
            _ => unreachable!("Snake body cannot be diagonal."),
        },
        _ => None,
    }
}

/// 0: Up, 1: Right, 2: Down, 3: Left.
fn snake_true_move(snake: &Snake, relative_move: isize) -> isize {
    let facing = snake_facing(snake).unwrap_or(0);
    (relative_move + facing).rem_euclid(4)
}

/// 0: Up, 1: Right, 2: Down, 3: Left.
fn snake_relative_move(snake: &Snake, true_move: isize) -> isize {
    let facing = snake_facing(snake).unwrap_or(0);
    (true_move - facing).rem_euclid(4)
}

pub fn snake_true_move2direction(snake_true_move: isize) -> Direction {
    match snake_true_move {
        UP => Direction::Up,
        RIGHT => Direction::Right,
        DOWN => Direction::Down,
        LEFT => Direction::Left,
        _ => unreachable!("`snake_true_move` should be 0 ~ 3."),
    }
}

fn fresh_game<R: RngCore>(rng: &mut R) -> Game {
    init_game(BOARD_SIZE, BOARD_SIZE, N_SNAKES, rng)
}

/// $R_e = f_e(101 - h)$.
fn eating_reward(eating_reward_factor: f64, health: u8) -> f64 {
    ((101 - health) as f64) * eating_reward_factor
}

const FOOD_RATE: f64 = 0.15;

/// Prints a message.
#[pyfunction]
fn hello() -> PyResult<String> {
    Ok("Hello from battlesnake_gym!".into())
}

/// A Python module implemented in Rust.
#[pymodule]
fn _lowlevel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::builder()
                .with_env_var("LOGLEVEL")
                .with_default_directive(Level::INFO.into())
                .from_env_lossy(),
        )
        .try_init();
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_class::<SnakeGame>()?;
    Ok(())
}

#[cfg(test)]
mod tests;
