use std::{
    cmp::Ordering::{Equal, Greater, Less},
    ops::Range,
};

use itertools::Itertools;
use numpy::{ndarray::Array3, IntoPyArray, PyArray3};
use pyo3::prelude::*;
use rand::{rngs::SmallRng, seq::IteratorRandom, Rng, SeedableRng};
use snork::{
    env::{Direction, Vec2D},
    game::{Game, Snake},
    grid::CellT,
    simulate::init_game,
};

mod observations;

use observations::*;

const BOARD_SIZE: usize = 11;
const N_SNAKES: usize = 4;
const WIN_REWARD: f64 = 1.0;
const LOSE_REWARD: f64 = -1.0;
const SURVIVE_ROUND_REWARD: f64 = 0.002;

// Clockwise, opposite to `np.rot90`.
const UP: isize = 0;
const RIGHT: isize = 1;
const DOWN: isize = 2;
const LEFT: isize = 3;

const PADDED_SIZE: usize = BOARD_SIZE * 2 - 1;
const N_LAYERS: usize = 9;
const HEALTH_NORMALIZATION: f64 = 1.0 / 239.0;

type BoundArray3<'py> = Bound<'py, PyArray3<f64>>;

#[pyclass]
#[derive(Clone, Debug)]
struct SnakeGame {
    pub game: Game,
    pub rng: SmallRng,
    pub food_count: u16,
}

#[pymethods]
impl SnakeGame {
    #[new]
    fn new() -> PyResult<Self> {
        let mut rng = SmallRng::from_entropy();
        let game = fresh_game(&mut rng);
        Ok(Self {
            game,
            rng,
            food_count: N_SNAKES as u16,
        })
    }

    fn reset(&mut self) -> PyResult<()> {
        self.game = fresh_game(&mut self.rng);
        self.food_count = N_SNAKES as u16;
        Ok(())
    }

    fn step(
        &mut self,
        actions: [isize; N_SNAKES],
    ) -> PyResult<([f64; N_SNAKES], [bool; N_SNAKES])> {
        let alive_before: Vec<_> = self.game.snakes.iter().map(|s| s.alive()).collect();

        // Based on `snork/src/simulate.rs`.
        let mut moves = [Direction::Up; N_SNAKES];
        for (index, &relative_move) in actions.iter().enumerate() {
            moves[index] = match snake_true_move(&self.game.snakes[index], relative_move) {
                UP => Direction::Up,
                RIGHT => Direction::Right,
                DOWN => Direction::Down,
                LEFT => Direction::Left,
                _ => unreachable!("`snake_true_move` is 0 ~ 3."),
            };
        }
        self.game.step(&moves);

        // Based on `snork/src/game.rs`.
        let mut n_alive = 0usize;
        let mut rewards = [0.0; N_SNAKES];
        let mut terminations = [false; N_SNAKES];
        let mut survivor = 0;
        for (i, snake) in self.game.snakes.iter().enumerate() {
            if snake.alive() {
                // Stayed alive.
                n_alive += 1;
                survivor = i;
                rewards[i] = SURVIVE_ROUND_REWARD;
            } else if alive_before[i] {
                // Died this round.
                rewards[i] = LOSE_REWARD;
                terminations[i] = true;
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
                for snake in &self.game.snakes {
                    if snake.alive() && snake.health == 100 {
                        self.food_count -= 1;
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

    fn snake_true_move(&self, snake_index: usize, relative_move: isize) -> PyResult<isize> {
        Ok(snake_true_move(
            &self.game.snakes[snake_index],
            relative_move,
        ))
    }
}

fn snake_facing(snake: &Snake) -> Option<isize> {
    let mut bodys = snake.body.iter();
    match (bodys.next(), bodys.next()) {
        (Some(tail), Some(body)) => match (tail.x.cmp(&body.x), tail.y.cmp(&body.y)) {
            (Less, Equal) => Some(RIGHT),
            (Equal, Less) => Some(UP),
            (Equal, Equal) => None,
            (Equal, Greater) => Some(DOWN),
            (Greater, Equal) => Some(LEFT),
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

fn fresh_game(rng: &mut SmallRng) -> Game {
    init_game(BOARD_SIZE, BOARD_SIZE, N_SNAKES, rng)
}

const FOOD_RATE: f64 = 0.15;

/// Prints a message.
#[pyfunction]
fn hello() -> PyResult<String> {
    Ok("Hello from battlesnake-gym!".into())
}

/// A Python module implemented in Rust.
#[pymodule]
fn _lowlevel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_class::<SnakeGame>()?;
    Ok(())
}

#[cfg(test)]
mod tests;
