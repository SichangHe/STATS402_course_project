use pyo3::{exceptions::PyValueError, prelude::*};
use rand::{rngs::SmallRng, seq::IteratorRandom, Rng, SeedableRng};
use snork::{
    env::{v2, Direction},
    game::{Game, Outcome, Snake},
    grid::CellT,
    simulate::init_game,
};

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
            food_count: 4,
        })
    }

    fn reset(&mut self) -> PyResult<()> {
        self.game = fresh_game(&mut self.rng);
        self.food_count = 4;
        Ok(())
    }

    fn step(&mut self, actions: [isize; 4]) -> PyResult<()> {
        // Based on `snork/src/simulate.rs`.
        let mut moves = [Direction::Up; 4];
        for (index, action) in actions.iter().enumerate() {
            // TODO: Implement rotation.
            moves[index] = match action {
                0 => Direction::Up,
                1 => Direction::Down,
                2 => Direction::Left,
                3 => Direction::Right,
                _ => return Err(PyValueError::new_err("Invalid action number")),
            };
        }
        self.game.step(&moves);

        let outcome = self.game.outcome();
        if outcome == Outcome::None {
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

        todo!()
    }

    fn render(&self) -> PyResult<()> {
        todo!()
    }
}

fn fresh_game(rng: &mut SmallRng) -> Game {
    init_game(11, 11, 4, rng)
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
