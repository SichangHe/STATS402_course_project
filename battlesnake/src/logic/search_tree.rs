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

    pub async fn compute_next_layer(&mut self, model: &Model) -> Result<Direction> {
        let new_leaf_nodes = self
            .leaf_nodes
            .iter()
            .copied()
            .map(|leaf_index| self.expand_leaf_node(leaf_index, model));
        let new_leaf_nodes = Vec::with_capacity(self.leaf_nodes.len() * 4);
        // TODO: Make parallel.
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
    ) -> Result<ArrayVec<[Vec<([Direction; 4], SearchTreeNode<'a>)>; 3]>> {
        let leaf_node = self.get_node(leaf_index);
        let actions = &leaf_node.probable_actions;
        let opponent_action_combos = actions[1]
            .iter()
            .cartesian_product(&actions[2])
            .cartesian_product(&actions[3]);
        // TODO: Make parallel.
        // TODO: Alpha-Beta pruning (referencing `prune_leaf_nodes`).
        let your_action_and_children = stream::iter(&actions[0])
            .then(|d0| {
                stream::iter(opponent_action_combos.clone())
                    .then(|((d1, d2), d3)| async {
                        let actions = [*d0, *d1, *d2, *d3];
                        let mut game = leaf_node.game.clone();
                        game.step(&actions);
                        match game.outcome() {
                            Outcome::None => make_node(game, model, self.depth)
                                .await
                                .map(|node| (actions, node)),
                            Outcome::Winner(winner) => Ok((
                                actions,
                                SearchTreeNode::terminal(game, self.depth, Some(winner)),
                            )),
                            Outcome::Match => {
                                Ok((actions, SearchTreeNode::terminal(game, self.depth, None)))
                            }
                        }
                    })
                    .try_collect::<Vec<_>>()
            })
            .try_collect::<ArrayVec<_>>()
            .await?;

        Ok(your_action_and_children)
    }

    fn insert_node(&mut self, node: SearchTreeNode<'a>) -> SearchTreeIndex<'a> {
        let index = self.nodes.len();
        self.nodes.push(node);
        SearchTreeIndex {
            index,
            _phantom: PhantomData,
        }
    }

    async fn make_node(&mut self, game: Game, model: &Model) -> Result<SearchTreeIndex<'a>> {
        let node = make_node(game, model, self.depth).await?;
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

async fn make_node<'a>(game: Game, model: &Model, depth: usize) -> Result<SearchTreeNode<'a>> {
    let prediction = model.predict(&game)?;
    let probable_actions = snake_probable_actions(&prediction);
    let node = SearchTreeNode {
        game,
        depth,
        rewards: prediction.values,
        probable_actions,
        children: Default::default(),
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
    pub children: ArrayVec<[SearchTreeChildIndex<'a>; 3]>,
}

impl<'a> SearchTreeNode<'a> {
    fn terminal(game: Game, depth: usize, winner: Option<u8>) -> Self {
        let mut rewards = [LOSE_REWARD; 4];
        if let Some(player_id) = winner {
            rewards[player_id as usize] = WIN_REWARD;
        }
        Self {
            game,
            depth,
            rewards,
            probable_actions: Default::default(),
            children: Default::default(),
        }
    }
}

#[derive(Clone, Debug, Default)]
struct SearchTreeChild<'a> {
    pub your_action: Direction,
    pub opponent_action_and_nodes: ArrayVec<[([Direction; 3], SearchTreeIndex<'a>); 3]>,
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
