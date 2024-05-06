use std::collections::HashMap;

use super::*;

#[derive(Clone, Debug)]
pub struct SearchTree<'a> {
    pub depth: usize,
    pub nodes: Vec<SearchTreeNode<'a>>,
    pub children: Vec<SearchTreeChild<'a>>,
    pub leaf_nodes: Vec<SearchTreeIndex<'a>>,
}

impl<'a> SearchTree<'a> {
    pub async fn try_new(game: Game, model: &Model) -> Result<Self> {
        let mut result = Self {
            depth: 0,
            nodes: Vec::with_capacity(1024),
            children: Vec::with_capacity(512),
            leaf_nodes: Vec::with_capacity(256),
        };

        let root_index = result.make_node(None, game, model).await?;
        result.leaf_nodes.push(root_index);

        Ok(result)
    }

    pub fn root(&self) -> &SearchTreeNode<'a> {
        &self.nodes[0]
    }

    pub fn get_node(&self, index: SearchTreeIndex<'a>) -> &SearchTreeNode<'a> {
        &self.nodes[index.index]
    }

    pub fn get_node_mut(&mut self, index: SearchTreeIndex<'a>) -> &mut SearchTreeNode<'a> {
        &mut self.nodes[index.index]
    }

    pub fn get_child(&self, index: SearchTreeChildIndex<'a>) -> &SearchTreeChild<'a> {
        &self.children[index.index]
    }

    pub fn get_child_mut(&mut self, index: SearchTreeChildIndex<'a>) -> &mut SearchTreeChild<'a> {
        &mut self.children[index.index]
    }

    /// Returns if there are more layers.
    pub async fn compute_next_layer(&mut self, model: &Model) -> Result<bool> {
        let mut leaf_node_new_children = Vec::with_capacity(self.leaf_nodes.len());
        trace!(?self.leaf_nodes);
        // TODO: Make parallel.
        for &leaf_index in &self.leaf_nodes {
            let leaf_node = self.get_node(leaf_index);
            if leaf_node.rewards[0] == WIN_REWARD || leaf_node.rewards[0] == LOSE_REWARD {
                continue; // Terminal nodes do not have children.
            }
            let children = expand_leaf_node(leaf_node, model, self.depth).await?;
            leaf_node_new_children.push((leaf_index, children));
        }
        trace!(leaf_node_new_children_len_before_pruning = leaf_node_new_children.len());
        prune_node_new_children(&mut leaf_node_new_children, self.leaf_nodes.len());
        let leaf_node_new_children = leaf_node_new_children;
        trace!(leaf_node_new_children_len = leaf_node_new_children.len());
        if leaf_node_new_children.is_empty() {
            // No leaf nodes to expand.
            return Ok(false);
        }
        // TODO: Alpha-Beta Pruning.

        self.leaf_nodes.clear();
        // Reserve space for new nodes and children.
        let n_new_nodes = leaf_node_new_children
            .iter()
            .map(|(_, children)| {
                children
                    .iter()
                    .map(|child| child.opponent_action_and_nodes.len())
                    .sum::<usize>()
            })
            .sum();
        self.nodes.reserve(n_new_nodes);
        self.leaf_nodes.reserve(n_new_nodes);
        let n_new_children = leaf_node_new_children
            .iter()
            .map(|(_, children)| children.len())
            .sum();
        self.children.reserve(n_new_children);
        debug!(
            node = self.nodes.capacity(),
            child = self.children.capacity(),
            leaf_node = self.leaf_nodes.capacity(),
            "Capacities reserved.",
        );
        let _break_future = async {}.await;

        for (leaf_index, children) in leaf_node_new_children {
            for OwnedChild {
                your_action,
                opponent_action_and_nodes,
                min_reward,
            } in children
            {
                let opponent_action_and_nodes = opponent_action_and_nodes
                    .into_iter()
                    .map(|(actions, node)| {
                        let node_index = self.insert_node(node);
                        self.leaf_nodes.push(node_index);
                        (actions, node_index)
                    })
                    .collect::<Vec<_>>();
                let node_indexes = opponent_action_and_nodes
                    .iter()
                    .map(|(_, node_index)| *node_index)
                    .collect::<Vec<_>>();
                let child = SearchTreeChild {
                    parent_node_index: leaf_index,
                    your_action,
                    opponent_action_and_nodes,
                    min_reward,
                };
                let child_index = self.insert_child(child);
                // Come back and update nodes' children and parent indexes.
                self.get_node_mut(leaf_index).children.push(child_index);
                for node_index in node_indexes {
                    let node = self.get_node_mut(node_index);
                    node.parent_child_index = Some(child_index);
                }
                let _break_future = async {}.await;
            }
        }

        self.back_propagate_rewards();
        self.depth += 1;
        debug!(
            ?self.depth,
            n_node = self.nodes.len(),
            n_child = self.children.len(),
            n_leaf_node = self.leaf_nodes.len(),
        );
        Ok(true)
    }

    fn back_propagate_rewards(&mut self) {
        for raw_node_index in (0..self.nodes.len()).rev() {
            let maybe_updated_reward = self.nodes[raw_node_index]
                .children
                .iter()
                .map(|&child_index| self.get_child(child_index).min_reward)
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            if let Some(updated_reward) = maybe_updated_reward {
                self.nodes[raw_node_index].rewards[0] = updated_reward;
            }
            let reward = self.nodes[raw_node_index].rewards[0];
            let parent_child_index = self.nodes[raw_node_index].parent_child_index;
            if let Some(parent_child_index) = parent_child_index {
                let parent_child = self.get_child_mut(parent_child_index);
                parent_child.min_reward = parent_child.min_reward.min(reward);
            }
        }
    }

    pub fn best_direction(&self) -> Option<Direction> {
        self.root()
            .children
            .iter()
            .copied()
            .map(|child_index| self.get_child(child_index))
            .max_by(|child0, child1| {
                child0
                    .min_reward
                    .partial_cmp(&child1.min_reward)
                    .unwrap_or(Ordering::Equal)
            })
            .map(|child| child.your_action)
    }

    fn insert_node(&mut self, node: SearchTreeNode<'a>) -> SearchTreeIndex<'a> {
        let index = self.nodes.len();
        self.nodes.push(node);
        SearchTreeIndex {
            index,
            _phantom: PhantomData,
        }
    }

    async fn make_node(
        &mut self,
        parent_child_index: Option<SearchTreeChildIndex<'a>>,
        game: Game,
        model: &Model,
    ) -> Result<SearchTreeIndex<'a>> {
        let node = make_node(parent_child_index, game, model, self.depth).await?;
        Ok(self.insert_node(node))
    }

    fn insert_child(&mut self, child: SearchTreeChild<'a>) -> SearchTreeChildIndex<'a> {
        let index = self.children.len();
        self.children.push(child);
        SearchTreeChildIndex {
            index,
            _phantom: PhantomData,
        }
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

async fn make_node<'a>(
    parent_child_index: Option<SearchTreeChildIndex<'a>>,
    game: Game,
    model: &Model,
    depth: usize,
) -> Result<SearchTreeNode<'a>> {
    let (game, prediction) = model.predict(game).await?;
    let prediction = prediction?;
    let mut action_probabilities = [[0.0; 4]; 4];
    for (player_id, policy_logit) in prediction.policy_logits.iter().enumerate() {
        let mut sum = 0.0;
        for (action_index, &logit) in policy_logit.iter().enumerate() {
            let raw_prob = logit.exp();
            action_probabilities[player_id][action_index] = raw_prob;
            sum += raw_prob;
        }
        for action_index in 0..4 {
            action_probabilities[player_id][action_index] /= sum;
        }
    }
    let node = SearchTreeNode {
        parent_child_index,
        game,
        depth,
        rewards: prediction.values,
        action_probabilities,
        children: Default::default(),
    };
    Ok(node)
}

const DIRECTIONS: [Direction; 4] = [
    Direction::Up,
    Direction::Right,
    Direction::Down,
    Direction::Left,
];

async fn expand_leaf_node(
    leaf_node: &SearchTreeNode<'_>,
    model: &Model,
    depth: usize,
) -> Result<ArrayVec<[OwnedChild; 3]>> {
    let action_probs = &leaf_node.action_probabilities;
    let mut three_quaters_max_action_prob = 0.75;
    for action_prob in action_probs {
        let max_prob = action_prob
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap_or(f64::MIN);
        three_quaters_max_action_prob *= max_prob;
    }

    // TODO: Make parallel.
    // TODO: Alpha-Beta pruning (referencing `prune_leaf_nodes`).
    let mut your_action_and_children = ArrayVec::<[OwnedChild; 3]>::default();
    for (d0, &prob0) in DIRECTIONS.into_iter().zip(&action_probs[0]) {
        if prob0 <= 0.0 {
            continue;
        }
        let mut opponent_action_and_nodes = Vec::with_capacity(27);
        let mut max_rewards = [f64::MIN; 4];
        for (d1, &prob1) in DIRECTIONS.into_iter().zip(&action_probs[1]) {
            if prob1 <= 0.0 {
                continue;
            }
            for (d2, &prob2) in DIRECTIONS.into_iter().zip(&action_probs[2]) {
                if prob2 <= 0.0 {
                    continue;
                }
                for (d3, &prob3) in DIRECTIONS.into_iter().zip(&action_probs[3]) {
                    if prob3 <= 0.0 {
                        continue;
                    }
                    if prob0 * prob1 * prob2 * prob3 < three_quaters_max_action_prob {
                        continue;
                    }
                    let mut game = leaf_node.game.clone();
                    game.step(&[d0, d1, d2, d3]);

                    let node = match game.outcome() {
                        Outcome::None => make_node(None, game, model, depth).await?,
                        Outcome::Winner(winner) => {
                            SearchTreeNode::terminal(None, game, depth, Some(winner))
                        }
                        Outcome::Match => SearchTreeNode::terminal(None, game, depth, None),
                    };

                    for (player_id, &reward) in node.rewards.iter().enumerate() {
                        max_rewards[player_id] = max_rewards[player_id].max(reward);
                    }
                    opponent_action_and_nodes.push(([d1, d2, d3], node));
                }
            }
        }

        for max_reward in max_rewards.iter_mut() {
            *max_reward = if *max_reward > 0.0 {
                *max_reward * 0.75
            } else {
                *max_reward / 0.75
            };
        }
        // Assume that the opponent will try to maximize their rewards.
        opponent_action_and_nodes.retain(|(_, node)| {
            node.rewards[1] >= max_rewards[1]
                && node.rewards[2] >= max_rewards[2]
                && node.rewards[3] >= max_rewards[3]
        });
        opponent_action_and_nodes.shrink_to_fit();

        // Compute the minimum reward after having the opponents choose
        // desirable actions.
        let min_reward = opponent_action_and_nodes
            .iter()
            .map(|(_, node)| node.rewards[0])
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap_or(f64::MIN);
        let child = OwnedChild {
            your_action: d0,
            opponent_action_and_nodes,
            min_reward,
        };
        your_action_and_children.push(child);
    }

    Ok(your_action_and_children)
}

/// Prunes leaf nodes that are not at least giving half the maximum reward.
fn prune_node_new_children(
    leaf_node_new_children: &mut Vec<(SearchTreeIndex<'_>, ArrayVec<[OwnedChild; 3]>)>,
    n_prev_leaf_node: usize,
) {
    if leaf_node_new_children.is_empty() {
        return;
    }
    let mut node_max_rewards = HashMap::with_capacity(n_prev_leaf_node);
    for (parent_index, children) in &*leaf_node_new_children {
        let entry = node_max_rewards.entry(*parent_index).or_insert(f64::MIN);
        let max_reward = children
            .iter()
            .map(|child| child.min_reward)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap_or(f64::MIN);
        trace!(?parent_index, max_reward);
        *entry = entry.max(max_reward);
    }
    let half_max_reward = node_max_rewards
        .iter()
        .map(|(_, &reward)| reward)
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .map(|max_reward| {
            if max_reward >= 0.0 {
                max_reward / 2.0
            } else {
                max_reward * 2.0
            }
        })
        .unwrap_or(f64::MIN);
    trace!(half_max_reward);

    leaf_node_new_children
        .retain(|(parent_index, _)| node_max_rewards[parent_index] >= half_max_reward);
}

#[derive(Clone, Debug)]
pub struct SearchTreeNode<'a> {
    pub parent_child_index: Option<SearchTreeChildIndex<'a>>,
    pub game: Game,
    pub depth: usize,
    /// Currently we keep rewards for all players, but only yours is used.
    pub rewards: [f64; 4],
    pub action_probabilities: [[f64; 4]; 4],
    pub children: ArrayVec<[SearchTreeChildIndex<'a>; 3]>,
}

impl<'a> SearchTreeNode<'a> {
    fn terminal(
        parent_child_index: Option<SearchTreeChildIndex<'a>>,
        game: Game,
        depth: usize,
        winner: Option<u8>,
    ) -> Self {
        let mut rewards = [LOSE_REWARD; 4];
        if let Some(player_id) = winner {
            rewards[player_id as usize] = WIN_REWARD;
        }
        Self {
            parent_child_index,
            game,
            depth,
            rewards,
            action_probabilities: Default::default(),
            children: Default::default(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SearchTreeChild<'a> {
    pub parent_node_index: SearchTreeIndex<'a>,
    pub your_action: Direction,
    pub opponent_action_and_nodes: Vec<([Direction; 3], SearchTreeIndex<'a>)>,
    pub min_reward: f64,
}

#[derive(Clone, Debug, Default)]
struct OwnedChild {
    pub your_action: Direction,
    pub opponent_action_and_nodes: Vec<([Direction; 3], SearchTreeNode<'static>)>,
    pub min_reward: f64,
}

#[derive(Copy)]
#[derive_everything]
pub struct SearchTreeIndex<'a> {
    index: usize,
    _phantom: PhantomData<&'a ()>,
}

#[derive(Copy)]
#[derive_everything]
pub struct SearchTreeChildIndex<'a> {
    index: usize,
    _phantom: PhantomData<&'a ()>,
}
