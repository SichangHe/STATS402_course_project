use std::{cmp::Ordering, f64::consts::LN_2};

use battlesnake_gym::snake_true_move2direction;
use model::*;
use shame::derive_everything;

use super::*;

#[derive(Clone, Debug)]
pub struct SearchTree<'a> {
    pub depth: usize,
    pub nodes: Vec<SearchTreeNode<'a>>,
    pub leaf_nodes: Vec<SearchTreeIndex<'a>>,
    pub _phantom: PhantomData<&'a ()>,
}

impl<'a> SearchTree<'a> {
    pub fn new(game: Game) -> Self {
        let mut result = Self {
            depth: 0,
            nodes: Vec::with_capacity(1024),
            leaf_nodes: Vec::with_capacity(256),
            _phantom: PhantomData,
        };

        let root_index = result.make_node(game, 0, [Direction::Up; 4], 0.0);
        result.leaf_nodes.push(root_index);

        result
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

    pub fn compute_next_layer(&mut self, model: &mut Model) -> Result<Direction> {
        let new_leaf_nodes = Vec::with_capacity(self.leaf_nodes.len() * 4);
        for leaf_index in &self.leaf_nodes {
            let leaf_node = self.get(*leaf_index);
            // TODO: Early break.

            let prediction = model.predict(&leaf_node.game)?;
            let probable_actions = snake_probable_actions(&prediction, &leaf_node.game);

            // TODO: store probable actions & values in leaf nodes.
            // TODO: Update list of leaf nodes.
        }
        // TODO: Prune unlikely leaf nodes.
        self.leaf_nodes = new_leaf_nodes;

        todo!()
    }

    fn insert(&mut self, node: SearchTreeNode<'a>) -> SearchTreeIndex<'a> {
        let index = self.nodes.len();
        self.nodes.push(node);
        SearchTreeIndex {
            index,
            _phantom: PhantomData,
        }
    }

    fn make_node(
        &mut self,
        game: Game,
        player_id: usize,
        actions: [Direction; 4],
        reward: f64,
    ) -> SearchTreeIndex<'a> {
        let node = SearchTreeNode {
            game,
            depth: self.depth,
            player_id,
            actions,
            reward,
            children: Vec::new(),
            _phantom: PhantomData,
        };
        self.insert(node)
    }
}

pub fn snake_probable_actions(prediction: &Prediction, game: &Game) -> [[Option<Direction>; 3]; 4] {
    let mut probable_actions = [[None; 3]; 4];
    for (player_id, &action_logits) in prediction.policy_logits.iter().enumerate() {
        let max_logit = action_logits
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap();
        action_logits
            .iter()
            .enumerate()
            .filter(|(_, &logit)| {
                // TODO:
                // This is equivalent to probability being at least one half the
                // maximum probability.,
                max_logit - logit <= LN_2
            })
            .enumerate()
            .for_each(|(probable_action_index, (action_index, _))| {
                probable_actions[player_id][probable_action_index] =
                    Some(snake_true_move2direction(action_index as isize));
            });
    }
    probable_actions
}

#[derive(Clone, Debug)]
pub struct SearchTreeNode<'a> {
    pub game: Game,
    pub depth: usize,
    pub player_id: usize,
    pub actions: [Direction; 4],
    pub reward: f64,
    pub children: Vec<SearchTreeIndex<'a>>,
    pub _phantom: PhantomData<&'a ()>,
}

#[derive(Copy)]
#[derive_everything]
pub struct SearchTreeIndex<'a> {
    pub index: usize,
    pub _phantom: PhantomData<&'a ()>,
}
