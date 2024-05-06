use tokio::{pin, select, sync::mpsc, time};

use model::*;

use super::*;

mod search_tree;

use search_tree::*;

const NO_TIME: Duration = Duration::from_secs(0);

pub async fn respond_move(game: &Game, timeout: Duration) -> Result<MoveResponse> {
    let direction = make_move(game, timeout).await?;
    info!(?direction);
    Ok(MoveResponse::new(direction))
}

#[instrument(skip(game))]
async fn make_move(game: &Game, timeout: Duration) -> Result<Direction> {
    debug!(?game);
    let (sender, mut receiver) = mpsc::channel(8);

    let searches = tree_searches(game, sender);
    let searches = time::timeout(timeout, async move {
        match searches.await {
            Ok(()) => {}
            Err(why) => error!(?why, "tree_searches"),
        }
    });
    pin!(searches);

    let mut direction = None;
    while let Some(new_direction) = select! {
        new_direction = receiver.recv() => new_direction,
        _ = &mut searches => None,
    } {
        direction = Some(new_direction);
        debug!(?new_direction);
    }

    let direction = direction.context("Tree search did not return a direction")?;
    info!(?direction);
    Ok(direction)
}

/// To avoid out of RAM.
const TOO_MANY_NODES: usize = 0x100_000;

async fn tree_searches(game: &Game, sender: mpsc::Sender<Direction>) -> Result<()> {
    let model = Model::new();
    let mut search_tree = SearchTree::try_new(game.clone(), &model).await?;

    while search_tree.nodes.len() < TOO_MANY_NODES {
        let new_direction = search_tree.compute_next_layer(&model).await?;
        sender.send(new_direction).await?;
        debug!(?new_direction, "Sent");
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
