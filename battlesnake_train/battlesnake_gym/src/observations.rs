use super::*;

pub fn states(game: &Game) -> (Vec<Array3<f64>>, Vec<isize>) {
    let body_layers = body_layers(game);
    let snake_order = sort_snakes(game);
    let head_values = head_values(game, &snake_order);
    trace!(?body_layers);
    debug!(?snake_order, ?head_values);

    let states = (0..N_SNAKES)
        .map(|snake_index| state(game, snake_index, &body_layers, &snake_order, &head_values))
        .collect::<Vec<_>>();

    let snake_facings = (0..N_SNAKES)
        .map(|snake_index| snake_facing(&game.snakes[snake_index]).unwrap_or(0))
        .collect::<Vec<_>>();

    (states, snake_facings)
}

/// $v_b:=\frac{2L_{\text{rest of body}}-1}{239}$
fn body_layers(game: &Game) -> Array3<f64> {
    let mut body_layers = Array3::<f64>::zeros((N_SNAKES, BOARD_SIZE, BOARD_SIZE));
    for (i, snake) in game.snakes.iter().enumerate() {
        for (rest_len, &Vec2D { x, y }) in snake.body.iter().enumerate() {
            let (rest_len, x, y) = (rest_len as f64, x as usize, y as usize);
            body_layers[[i, x, y]] = rest_len * 2.0 - 1.0;
        }
    }
    body_layers *= HEALTH_NORMALIZATION;
    body_layers
}

fn sort_snakes(game: &Game) -> [usize; N_SNAKES] {
    let mut snakes: Vec<_> = game
        .snakes
        .iter()
        .enumerate()
        .map(|(i, snake)| (snake.body.len(), snake.health, i))
        .collect();
    snakes.sort_unstable();
    let mut order = [0; N_SNAKES];
    for (i, (_, _, snake_index)) in snakes.iter().enumerate() {
        order[i] = *snake_index;
    }
    order
}

/// $v_h:=\frac{1+2(L_{\text{opponent}}-L_{\text{us}})}{239}$
fn head_values(game: &Game, snake_order: &[usize]) -> [[f64; N_SNAKES - 1]; N_SNAKES] {
    let mut head_values = [[0.0; N_SNAKES - 1]; N_SNAKES];
    for (you_index, you) in game.snakes.iter().enumerate() {
        for (opponent_index, &snake_index) in snake_order
            .iter()
            .filter(|their_index| **their_index != you_index)
            .enumerate()
        {
            let opponent = &game.snakes[snake_index];
            let len_diff = opponent.body.len() as f64 - you.body.len() as f64;
            let value = (2.0 * len_diff + 1.0) * HEALTH_NORMALIZATION;
            head_values[you_index][opponent_index] = value;
        }
    }
    head_values
}

/// Layers:
/// - 0: Walls.
/// - 1: Your body.
/// - 2, 4, 6: Opponent head.
/// - 3, 5, 7: Opponent body.
/// - 8: Food.
fn state(
    game: &Game,
    you_index: usize,
    body_layers: &Array3<f64>,
    snake_order: &[usize],
    head_values: &[[f64; N_SNAKES - 1]],
) -> Array3<f64> {
    let state = Array3::<f64>::zeros((N_LAYERS, PADDED_SIZE, PADDED_SIZE));
    let you = &game.snakes[you_index];

    let (head_x, head_y) = if let Some(&Vec2D {
        x: head_x,
        y: head_y,
    }) = you.body.iter().last()
    {
        (head_x as usize, head_y as usize)
    } else {
        return state;
    };
    debug!(you_index, head_x, head_y);
    // Based on `make_state` in `AlphaSnake-Zero/code/utils/game.py`.
    let dx = |x| x + (BOARD_SIZE - 1) - head_x;
    let dy = |y| y + (BOARD_SIZE - 1) - head_y;
    let mut state_ctx = StateCtx {
        state,
        game,
        you_index,
        body_layers,
        snake_order,
        head_values,
        dx,
        dy,
    };

    state_ctx.walls();
    state_ctx.your_body();
    state_ctx.opponents();
    state_ctx.food();

    state_ctx.state
}

struct StateCtx<'a, DX, DY>
where
    DX: Fn(usize) -> usize,
    DY: Fn(usize) -> usize,
{
    state: Array3<f64>,
    game: &'a Game,
    you_index: usize,
    body_layers: &'a Array3<f64>,
    snake_order: &'a [usize],
    head_values: &'a [[f64; N_SNAKES - 1]],
    dx: DX,
    dy: DY,
}

impl<'a, DX, DY> StateCtx<'a, DX, DY>
where
    DX: Fn(usize) -> usize,
    DY: Fn(usize) -> usize,
{
    // Do not inline, for easier debugging.
    #[inline(never)]
    fn walls(&mut self) {
        let (state, dx, dy) = (&mut self.state, &self.dx, &self.dy);
        for (x, y) in (0..PADDED_SIZE).cartesian_product(0..PADDED_SIZE) {
            state[[0, x, y]] = 1.0;
        }
        for (x, y) in board_indices() {
            state[[0, dx(x), dy(y)]] = 0.0;
        }
    }

    #[inline(never)]
    fn your_body(&mut self) {
        let (state, you_index, body_layers, dx, dy) = (
            &mut self.state,
            self.you_index,
            self.body_layers,
            &self.dx,
            &self.dy,
        );
        for (x, y) in board_indices() {
            state[[1, dx(x), dy(y)]] = body_layers[[you_index, x, y]];
        }
    }

    #[inline(never)]
    fn opponents(&mut self) {
        let (state, game, you_index, body_layers, snake_order, head_values, dx, dy) = (
            &mut self.state,
            self.game,
            self.you_index,
            self.body_layers,
            self.snake_order,
            self.head_values,
            &self.dx,
            &self.dy,
        );
        for (opponent_index, &snake_index) in snake_order
            .iter()
            .filter(|snake_index| **snake_index != you_index)
            .enumerate()
        {
            if let Some(&Vec2D { x, y }) = game.snakes[snake_index].body.iter().last() {
                let (x, y) = (x as usize, y as usize);
                state[[2 * opponent_index + 2, dx(x), dy(y)]] =
                    head_values[you_index][opponent_index];

                for (x, y) in board_indices() {
                    state[[2 * opponent_index + 3, dx(x), dy(y)]] =
                        body_layers[[snake_index, x, y]];
                }
            } // Else: this opponent is dead.
        }
    }

    #[inline(never)]
    fn food(&mut self) {
        let (state, game, you, dx, dy) = (
            &mut self.state,
            self.game,
            &self.game.snakes[self.you_index],
            &self.dx,
            &self.dy,
        );
        let food_value = food_value(you.health as usize);
        for (x, y) in board_indices() {
            let cell = &game.grid.cells[x + y * BOARD_SIZE];
            if let CellT::Food = cell.t {
                state[[8, dx(x), dy(y)]] = food_value;
            }
        }
    }
}

fn board_indices() -> itertools::Product<Range<usize>, Range<usize>> {
    (0..BOARD_SIZE).cartesian_product(0..BOARD_SIZE)
}

/// $\frac{101-H_{\text{us}}}{100}$
fn food_value(health: usize) -> f64 {
    (101 - health) as f64 / 100.0
}
