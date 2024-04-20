use numpy::ndarray::Array3;
use pyo3::{exceptions::PyValueError, prelude::*};
use rand::{rngs::SmallRng, seq::IteratorRandom, Rng, SeedableRng};
use snork::{env::Direction, game::Game, grid::CellT, simulate::init_game};

const BOARD_SIZE: usize = 11;
const N_SNAKES: usize = 4;
const WIN_REWARD: f32 = 1.0;
const LOSE_REWARD: f32 = -1.0;
const SURVIVE_ROUND_REWARD: f32 = 0.002;

const PADDED_SIZE: usize = BOARD_SIZE * 2 - 1;
const N_LAYERS: usize = 10;

#[pyclass]
#[derive(Clone, Debug)]
struct SnakeGame {
    game: Game,
    rng: SmallRng,
    food_count: u16,
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
    ) -> PyResult<([f32; N_SNAKES], [bool; N_SNAKES])> {
        let alive_before: Vec<_> = self.game.snakes.iter().map(|s| s.alive()).collect();

        // Based on `snork/src/simulate.rs`.
        let mut moves = [Direction::Up; N_SNAKES];
        for (index, action) in actions.iter().enumerate() {
            moves[index] = match action {
                0 => Direction::Up,
                1 => Direction::Down,
                2 => Direction::Left,
                3 => Direction::Right,
                _ => return Err(PyValueError::new_err("Invalid action number")),
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

    fn states(&self) {
        let mut states = Array3::<f32>::zeros((N_LAYERS, PADDED_SIZE, PADDED_SIZE));
        todo!()
    }

    fn render(&self) -> PyResult<()> {
        todo!()
    }
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
