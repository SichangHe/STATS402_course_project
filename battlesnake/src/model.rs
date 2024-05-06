use super::*;

pub struct Model {
    // TODO: Add fields
}

impl Model {
    pub fn new() -> Self {
        Self {}
    }

    /// Predict policy probabilities and values given observations.
    /// Invalid actions correspond to -inf logits.
    /// Invalid game states correspond to 0 values.
    /// Dead agents receive logits of 0 for only one of the actions.
    pub fn predict(&self, game: &Game) -> Result<Prediction> {
        let mut policy_logits = [[f64::NEG_INFINITY; 4]; 4];
        let mut values = [LOSE_REWARD; 4];
        // TODO: Use Python model.
        // TODO: Convert back to real move.

        // TODO: Below is a placeholder implementation.
        for (player_id, policy_logit) in policy_logits.iter_mut().enumerate() {
            if !game.snake_is_alive(player_id as u8) {
                policy_logit[UP as usize] = 0.0;
                continue;
            }
            for direction in snake_valid_directions(game, &(player_id as u8)) {
                let action_index = direction2snake_true_move(direction) as usize;
                policy_logit[action_index] = -LN_2; // Randomly give valid actions a 0.5 probability.
                values[action_index] = 0.0;
            }
        }

        Ok(Prediction {
            policy_logits,
            values,
        })
    }
}

fn snake_valid_directions<'a>(
    game: &'a Game,
    snake_id: &'a u8,
) -> impl Iterator<Item = Direction> + 'a {
    Direction::all()
        .into_iter()
        .filter(|dir| game.move_is_valid(*snake_id, *dir))
}

pub struct Prediction {
    pub policy_logits: [[f64; 4]; 4],
    pub values: [f64; 4],
}
