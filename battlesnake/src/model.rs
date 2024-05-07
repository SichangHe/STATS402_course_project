use battlesnake_gym::observations::states;
use numpy::convert::IntoPyArray;
use pyo3::{intern, types::IntoPyDict};
use tokio::{
    sync::Mutex,
    task::{JoinHandle, JoinSet},
};
use tokio_gen_server::*;

use super::*;

pub struct ModelActor {
    py_model: Arc<Py<PyAny>>,
    np_rot90: Arc<Py<PyAny>>,
    join_set: JoinSet<()>,
}

impl Actor for ModelActor {
    type CallMsg = Game;

    type CastMsg = ();

    type Reply = (Game, Result<Prediction>);

    async fn handle_call(
        &mut self,
        game: Self::CallMsg,
        _env: &mut Ref<Self>,
        reply_sender: tokio::sync::oneshot::Sender<Self::Reply>,
    ) -> Result<()> {
        // Clear join set.
        while let Some(prev_result) = self.join_set.try_join_next() {
            match prev_result {
                Ok(()) => {}
                Err(why) => error!(?why, "ModelActor join error."),
            }
        }
        let np_rot90 = Arc::clone(&self.np_rot90);
        let py_model = Arc::clone(&self.py_model);
        self.join_set.spawn(async move {
            let prediction = predict(&np_rot90, &py_model, &game);
            _ = reply_sender.send((game, prediction));
        });

        Ok(())
    }
}

/// Predict policy probabilities and values given observations.
/// Invalid actions correspond to -inf logits.
/// Invalid game states correspond to 0 values.
/// Dead agents receive logits of 0 for only one of the actions.
#[instrument(skip(np_rot90, py_model, game))]
pub fn predict(np_rot90: &Py<PyAny>, py_model: &Py<PyAny>, game: &Game) -> Result<Prediction> {
    trace!("Predicting.");
    let mut policy_logits = [[f64::NEG_INFINITY; 4]; 4];
    for (player_id, policy_logit) in policy_logits.iter_mut().enumerate() {
        if !game.snake_is_alive(player_id as u8) {
            policy_logit[0] = 0.0;
            continue;
        }
    }

    let mut values = [0.0; 4];
    let (raw_states, snake_facings) = states(game);

    Python::with_gil(|py| -> Result<_> {
        let observations = raw_states
            .into_iter()
            .zip(&snake_facings)
            .enumerate()
            .filter(|(player_id, _)| game.snake_is_alive(*player_id as u8))
            .map(|(_, (raw_state, &facing))| {
                let state = raw_state.into_pyarray_bound(py);
                let rot_kwargs = [(intern!(py, "k"), facing)].into_py_dict_bound(py);
                rot_kwargs.set_item(intern!(py, "axes"), (1, 2))?;
                let state = np_rot90.call_bound(py, (state,), Some(&rot_kwargs))?;
                Ok(state)
            })
            .collect::<Result<Vec<_>>>()?;
        let (raw_policy_logits, raw_values): (Vec<[f64; 3]>, Vec<f64>) = py_model
            .call_method1(py, intern!(py, "policy_logits_and_values"), (observations,))?
            .extract(py)?;
        (0..4)
            .filter(|player_id| game.snake_is_alive(*player_id as u8))
            .zip(raw_policy_logits)
            .zip(raw_values)
            .zip(&snake_facings)
            .for_each(|(((player_id, policy_logit), value), &snake_facing)| {
                (-1..2)
                    .zip(policy_logit)
                    .for_each(|(relative_move, logit)| {
                        let true_move = (relative_move + snake_facing).rem_euclid(4);
                        let action = snake_true_move2direction(true_move);
                        if game.move_is_valid(player_id as u8, action) {
                            policy_logits[player_id][true_move as usize] = logit;
                        }
                    });
                values[player_id] = value;
            });

        Ok(())
    })?;

    trace!("Predicted.");
    Ok(Prediction {
        policy_logits,
        values,
    })
}

pub struct Model {
    pub handle: JoinHandle<Result<()>>,
    pub actor_ref: Mutex<Ref<ModelActor>>,
}

impl Model {
    #[allow(non_snake_case)]
    #[instrument]
    pub fn try_new() -> PyResult<Self> {
        debug!("Loading model.");
        let actor = Python::with_gil(|py| -> PyResult<_> {
            let numpy = PyModule::import_bound(py, "numpy")?;
            let rot90 = numpy.getattr("rot90")?;

            let battlesnake_gym = PyModule::import_bound(py, "battlesnake_gym")?;
            let BattlesnakeEnv = battlesnake_gym.getattr("BattlesnakeEnv")?;
            let battlesnake_train = PyModule::import_bound(py, "battlesnake_train")?;
            let ppo = battlesnake_train.getattr("ppo")?;
            let DynPPO = ppo.getattr("DynPPO")?;

            let env = BattlesnakeEnv.call0()?;
            let model = DynPPO.call_method(
                "load_trial",
                (env,),
                Some(&[("save_model_name", "vit-tiny")].into_py_dict_bound(py)),
            )?;
            let trial_index = model.getattr("trial_index")?;
            info!(?trial_index);

            Ok(ModelActor {
                py_model: Arc::new(model.unbind()),
                np_rot90: Arc::new(rot90.unbind()),
                join_set: Default::default(),
            })
        })?;

        let (handle, actor_ref) = actor.spawn();
        let actor_ref = Mutex::new(actor_ref);
        Ok(Self { handle, actor_ref })
    }

    /// Predict policy probabilities and values given observations.
    /// Invalid actions correspond to -inf logits.
    /// Invalid game states correspond to 0 values.
    /// Dead agents receive logits of 0 for only one of the actions.
    #[instrument(skip(self, game))]
    pub async fn predict(&self, game: Game) -> Result<(Game, Result<Prediction>)> {
        self.actor_ref.lock().await.call(game).await
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
