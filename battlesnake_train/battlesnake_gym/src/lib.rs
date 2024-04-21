use std::cmp::Ordering::{Equal, Greater, Less};

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

    fn render(&self) -> PyResult<()> {
        todo!()
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
    (relative_move + facing) % 4
}

fn states(game: &Game) -> (Vec<Array3<f64>>, Vec<isize>) {
    let body_layers = body_layers(game);
    let snake_order = sort_snakes(game);
    let head_values = head_values(game, &snake_order);
    eprintln!(
        "Body layers: {body_layers:?}\nSnake order: {snake_order:?}\nHead values: {head_values:?}"
    );

    let states = (0..N_SNAKES)
        .map(|snake_index| state(game, snake_index, &body_layers, &snake_order, &head_values))
        .collect::<Vec<_>>();

    let snake_facings = (0..N_SNAKES)
        .map(|snake_index| snake_facing(&game.snakes[snake_index]).unwrap_or(0))
        .collect::<Vec<_>>();

    (states, snake_facings)
}

/// $v_b:=\frac{2L_{\text{rest of body}}-1}{239}$
fn body_layers(game: &Game) -> Array3<f64> {
    let mut body_layers = Array3::<f64>::zeros((N_SNAKES, BOARD_SIZE, BOARD_SIZE));
    for (i, snake) in game.snakes.iter().enumerate() {
        for (rest_len, &Vec2D { x, y }) in snake.body.iter().enumerate() {
            let (rest_len, x, y) = (rest_len as f64, x as usize, y as usize);
            body_layers[[i, x, y]] = rest_len * 2.0 - 1.0;
        }
    }
    body_layers *= HEALTH_NORMALIZATION;
    body_layers
}

fn sort_snakes(game: &Game) -> [usize; N_SNAKES] {
    let mut snakes: Vec<_> = game
        .snakes
        .iter()
        .enumerate()
        .map(|(i, snake)| (snake.body.len(), snake.health, i))
        .collect();
    snakes.sort_unstable();
    let mut order = [0; N_SNAKES];
    for (i, (_, _, snake_index)) in snakes.iter().enumerate() {
        order[i] = *snake_index;
    }
    order
}

/// $v_h:=\frac{1+2(L_{\text{opponent}}-L_{\text{us}})}{239}$
fn head_values(game: &Game, snake_order: &[usize]) -> [[f64; N_SNAKES - 1]; N_SNAKES] {
    let mut head_values = [[0.0; N_SNAKES - 1]; N_SNAKES];
    for (you_index, you) in game.snakes.iter().enumerate() {
        for (opponent_index, &snake_index) in snake_order
            .iter()
            .filter(|their_index| **their_index != you_index)
            .enumerate()
        {
            let opponent = &game.snakes[snake_index];
            let len_diff = opponent.body.len() as f64 - you.body.len() as f64;
            let value = (2.0 * len_diff + 1.0) * HEALTH_NORMALIZATION;
            head_values[you_index][opponent_index] = value;
        }
    }
    head_values
}

/// Layers:
/// - 0: Walls.
/// - 1: Your body.
/// - 2, 4, 6: Opponent head.
/// - 3, 5, 7: Opponent body.
/// - 8: Food.
fn state(
    game: &Game,
    you_index: usize,
    body_layers: &Array3<f64>,
    snake_order: &[usize],
    head_values: &[[f64; N_SNAKES - 1]],
) -> Array3<f64> {
    let mut state = Array3::<f64>::zeros((N_LAYERS, PADDED_SIZE, PADDED_SIZE));
    let you = &game.snakes[you_index];

    let (head_x, head_y) = if let Some(&Vec2D {
        x: head_x,
        y: head_y,
    }) = you.body.iter().last()
    {
        (head_x as usize, head_y as usize)
    } else {
        return state;
    };
    eprintln!(
        "Your index: {}, Your head: ({}, {})",
        you_index, head_x, head_y
    );
    let dx = |x| x + BOARD_SIZE - head_x;
    let dy = |y| y + BOARD_SIZE - head_y;
    let board_indices = (0..BOARD_SIZE).cartesian_product(0..BOARD_SIZE);
    // Walls.
    for (x, y) in (0..PADDED_SIZE).cartesian_product(0..PADDED_SIZE) {
        state[[0, x, y]] = 1.0;
    }
    for (x, y) in board_indices.clone() {
        state[[0, dx(x), dy(y)]] = 0.0;
    }
    // Your body.
    for (x, y) in board_indices.clone() {
        state[[1, dx(x), dy(y)]] = body_layers[[you_index, x, y]];
    }
    // Opponents.
    for (opponent_index, &snake_index) in snake_order
        .iter()
        .filter(|snake_index| **snake_index != you_index)
        .enumerate()
    {
        if let Some(&Vec2D { x, y }) = game.snakes[snake_index].body.iter().last() {
            let (x, y) = (x as usize, y as usize);
            state[[2 * opponent_index + 2, dx(x), dy(y)]] = head_values[you_index][opponent_index];

            for (x, y) in board_indices.clone() {
                state[[2 * opponent_index + 3, dx(x), dy(y)]] = body_layers[[snake_index, x, y]];
            }
        } // Else: this opponent is dead.
    }
    // Food.
    let food_value = food_value(you.health as usize);
    for (x, y) in board_indices {
        let cell = &game.grid.cells[x + y * BOARD_SIZE];
        if let CellT::Food = cell.t {
            state[[8, dx(x), dy(y)]] = food_value;
        }
    }

    state
}

/// $\frac{101-H_{\text{us}}}{100}$
fn food_value(health: usize) -> f64 {
    (101 - health) as f64 / 100.0
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
