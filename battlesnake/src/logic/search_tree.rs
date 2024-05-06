use std::{cmp::Ordering, f64::consts::LN_2};

use battlesnake_gym::{direction2snake_true_move, snake_true_move2direction};
use shame::derive_everything;
use tinyvec::ArrayVec;

use super::*;

#[derive(Clone, Debug)]
pub struct SearchTree<'a> {
    pub depth: usize,
    pub nodes: Vec<SearchTreeNode<'a>>,
    pub leaf_nodes: Vec<SearchTreeIndex<'a>>,
    pub _phantom: PhantomData<&'a ()>,
}

impl<'a> SearchTree<'a> {
    pub async fn try_new(game: Game, model: &Model) -> Result<Self> {
        let mut result = Self {
            depth: 0,
            nodes: Vec::with_capacity(1024),
            leaf_nodes: Vec::with_capacity(256),
            _phantom: PhantomData,
        };

        let root_index = result.make_node(game, model).await?;
        result.leaf_nodes.push(root_index);

        Ok(result)
    }

    pub fn root(&self) -> &SearchTreeNode<'a> {
        &self.nodes[0]
    }

    pub fn root_mut(&mut self) -> &mut SearchTreeNode<'a> {
        &mut self.nodes[0]
    }

    pub fn get(&self, index: SearchTreeIndex<'a>) -> &SearchTreeNode<'a> {
        &self.nodes[index.index]
    }

    pub fn get_mut(&mut self, index: SearchTreeIndex<'a>) -> &mut SearchTreeNode<'a> {
        &mut self.nodes[index.index]
    }

    pub async fn compute_next_layer(&mut self, model: &Model) -> Result<Direction> {
        let new_leaf_nodes = Vec::with_capacity(self.leaf_nodes.len() * 4);
        for &leaf_index in &self.leaf_nodes {
            let expansion = self.expand_leaf_node(leaf_index, model).await?;
            // TODO: Alpha-Beta Pruning.

            // TODO: Update list of leaf nodes.
            // TODO: Back-propagate rewards.
        }
        self.leaf_nodes = new_leaf_nodes;

        todo!()
    }

    async fn expand_leaf_node(
        &self,
        leaf_index: SearchTreeIndex<'a>,
        model: &Model,
    ) -> Result<Vec<([Direction; 4], SearchTreeNode<'a>)>> {
        let leaf_node = self.get(leaf_index);
        let actions = &leaf_node.probable_actions;
        let action_combos = actions[0]
            .iter()
            .cartesian_product(&actions[1])
            .cartesian_product(&actions[2])
            .cartesian_product(&actions[3]);
        // TODO: Make parallel.
        // TODO: Alpha-Beta pruning (referencing `prune_leaf_nodes`).
        let children = stream::iter(action_combos)
            .then(|(((d0, d1), d2), d3)| async {
                let game = leaf_node.game.clone();
                make_node(game, model, self.depth)
                    .await
                    .map(|node| ([*d0, *d1, *d2, *d3], node))
            })
            .try_collect::<Vec<_>>()
            .await?;

        Ok(children)
    }

    fn insert(&mut self, node: SearchTreeNode<'a>) -> SearchTreeIndex<'a> {
        let index = self.nodes.len();
        self.nodes.push(node);
        SearchTreeIndex {
            index,
            _phantom: PhantomData,
        }
    }

    async fn make_node(&mut self, game: Game, model: &Model) -> Result<SearchTreeIndex<'a>> {
        let node = make_node(game, model, self.depth).await?;
        Ok(self.insert(node))
    }
}

pub fn snake_probable_actions(prediction: &Prediction) -> [ArrayVec<[Direction; 3]>; 4] {
    let mut probable_actions = [ArrayVec::default(); 4];
    for (player_id, &action_logits) in prediction.policy_logits.iter().enumerate() {
        let max_logit = action_logits
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap();
        action_logits
            .iter()
            .enumerate()
            .filter(|(_, &logit)| {
                // This is equivalent to probability being at least one half the
                // maximum probability.,
                max_logit - logit <= LN_2
            })
            .for_each(|(action_index, _)| {
                probable_actions[player_id].push(snake_true_move2direction(action_index as isize));
            });
    }
    probable_actions
}

async fn make_node<'a>(game: Game, model: &Model, depth: usize) -> Result<SearchTreeNode<'a>> {
    let prediction = model.predict(&game)?;
    let probable_actions = snake_probable_actions(&prediction);
    let node = SearchTreeNode {
        game,
        depth,
        rewards: prediction.values,
        probable_actions,
        children: Vec::new(),
        _phantom: PhantomData,
    };
    Ok(node)
}

// TODO: Remove after implementing Alpha-Beta pruning.
async fn prune_leaf_nodes(nodes: &mut Vec<([Direction; 4], SearchTreeNode<'_>)>) {
    if nodes.is_empty() {
        return;
    }
    let mut your_max_reward = f64::MIN;
    let mut your_max_rewards = [f64::MIN; 4];
    for (actions, node) in nodes.iter() {
        let your_reward = node.rewards[0];
        if your_reward > your_max_reward {
            your_max_reward = your_reward;
        }
        let action_id = direction2snake_true_move(actions[0]) as usize;
        if your_reward > your_max_rewards[action_id] {
            your_max_rewards[action_id] = your_reward;
        }
    }
    let half_your_max_reward = your_max_reward / 2.0;
    nodes.retain(|(actions, _)| {
        let action_id = direction2snake_true_move(actions[0]) as usize;
        your_max_rewards[action_id] >= half_your_max_reward
    });
    todo!();
}

#[derive(Clone, Debug)]
pub struct SearchTreeNode<'a> {
    pub game: Game,
    pub depth: usize,
    pub rewards: [f64; 4],
    pub probable_actions: [ArrayVec<[Direction; 3]>; 4],
    pub children: Vec<([Direction; 4], SearchTreeIndex<'a>)>,
    pub _phantom: PhantomData<&'a ()>,
}

#[derive(Copy)]
#[derive_everything]
pub struct SearchTreeIndex<'a> {
    index: usize,
    _phantom: PhantomData<&'a ()>,
}
