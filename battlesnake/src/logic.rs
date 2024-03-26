use battlesnake_game_types::{
    compact_representation::CellNum,
    types::{
        HealthGettableGame, NeighborDeterminableGame, SnakeBodyGettableGame, YouDeterminableGame,
    },
};

use super::*;

// TODO: Improve.
pub fn make_move(game: &StandardCellBoard4Snakes11x11) -> Move {
    MoveEvaluation::evaluate_one(game).best()
}

const FOOD_REWARD: f64 = 16.0;
const DEATH_PENALTY: f64 = f64::NEG_INFINITY;

#[derive(Clone, Copy, Debug)]
struct MoveEvaluation {
    up: f64,
    down: f64,
    left: f64,
    right: f64,
}

impl Default for MoveEvaluation {
    fn default() -> Self {
        Self {
            up: DEATH_PENALTY,
            down: DEATH_PENALTY,
            left: DEATH_PENALTY,
            right: DEATH_PENALTY,
        }
    }
}

impl MoveEvaluation {
    fn best(&self) -> Move {
        [Move::Up, Move::Down, Move::Left, Move::Right]
            .into_iter()
            .zip([self.up, self.down, self.left, self.right])
            .filter(|(_, weight)| !weight.is_nan())
            .max_by(|(_, left_weight), (_, right_weight)| {
                left_weight.partial_cmp(right_weight).unwrap()
            })
            .unwrap_or_else(|| {
                warn!("No valid moves, going up.");
                (Move::Up, 0.0)
            })
            .0
    }

    fn evaluate_one(game: &StandardCellBoard4Snakes11x11) -> Self {
        let mut evaluation = Self::default();
        let my_id = game.you_id();
        if !game.is_alive(my_id) {
            return evaluation;
        }

        let my_body = game.get_snake_body_vec(my_id);
        let my_head = my_body[0];

        for (possible_move, pos) in game.possible_moves(&my_head) {
            let cell = game.cells()[pos.0.as_usize()];
            let weight = if cell.is_food() {
                FOOD_REWARD
            } else if cell.is_empty() {
                0.0
            } else {
                DEATH_PENALTY
            };
            match possible_move {
                Move::Up => evaluation.up = weight,
                Move::Down => evaluation.down = weight,
                Move::Left => evaluation.left = weight,
                Move::Right => evaluation.right = weight,
            }
        }

        evaluation
    }
}
