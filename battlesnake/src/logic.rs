use tokio::time;
use tokio_scoped::scope;

use model::*;

use super::*;

mod search_tree;

use search_tree::*;

const NO_TIME: Duration = Duration::from_secs(0);

pub async fn respond_move(
    game: &Game,
    timeout: Duration,
    model: Arc<Model>,
) -> Result<MoveResponse> {
    let direction = make_move(game, timeout, model).await?;
    info!(?direction);
    Ok(MoveResponse::new(direction))
}

#[instrument(skip(game, model))]
async fn make_move(game: &Game, timeout: Duration, model: Arc<Model>) -> Result<Direction> {
    info!(?game);

    let mut direction = None;
    scope(|scope| {
        let searches = tree_searches(game.clone(), &mut direction, model);
        let searches = time::timeout(timeout, searches);
        let searches = async move {
            match searches.await {
                Ok(Err(why)) => error!(?why, "tree_searches"),
                Ok(Ok(())) | Err(_) => {}
            }
        };
        scope.spawn(searches);
    });

    let direction = direction.context("Tree search did not return a direction")?;
    info!(?direction);
    Ok(direction)
}

/// To avoid out of RAM.
const TOO_MANY_NODES: usize = 0x8_000;

async fn tree_searches(
    game: Game,
    direction: &mut Option<Direction>,
    model: Arc<Model>,
) -> Result<()> {
    let mut search_tree = SearchTree::try_new(game, &model).await?;

    loop {
        if search_tree.nodes.len() >= TOO_MANY_NODES {
            warn!(
                n_search_tree_node = search_tree.nodes.len(),
                "Too many nodes"
            );
            break;
        }
        let more_layers_exist = search_tree.compute_next_layer(&model).await?;
        if let Some(new_direction) = search_tree.best_direction() {
            *direction = Some(new_direction);
            debug!(?new_direction, "Updated");
        }
        if !more_layers_exist {
            info!("Search complete");
            break;
        }
    }

    Ok(())
}

fn actions_with_replaced(
    actions: [Direction; 4],
    replace_index: usize,
    direction: Direction,
) -> [Direction; 4] {
    let mut new_actions = actions;
    new_actions[replace_index] = direction;
    new_actions
}
