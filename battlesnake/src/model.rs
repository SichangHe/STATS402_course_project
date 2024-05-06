use super::*;

pub struct Model {
    // TODO: Add fields
}

impl Model {
    pub fn new() -> Self {
        todo!()
    }

    /// Predict policy probabilities and values given observations.
    /// Invalid actions correspond to -inf logits.
    /// Invalid game states correspond to 0 values.
    pub fn predict(&self, game: &Game) -> Result<Prediction> {
        // TODO: Convert back to real move.
        todo!()
    }
}

pub struct Prediction {
    pub policy_logits: [[f64; 4]; 4],
    pub values: [f64; 4],
}
