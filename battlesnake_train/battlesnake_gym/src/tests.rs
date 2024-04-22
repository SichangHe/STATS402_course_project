use super::*;

#[test]
fn run_game_straight_up() {
    run_game_one_direction_and_straight(UP);
}

#[test]
fn run_game_straight_right() {
    run_game_one_direction_and_straight(RIGHT);
}

#[test]
fn run_game_straight_down() {
    run_game_one_direction_and_straight(DOWN);
}

#[test]
fn run_game_straight_left() {
    run_game_one_direction_and_straight(LEFT);
}

fn run_game_one_direction_and_straight(direction: isize) {
    init_tracing();
    let mut game = SnakeGame::new().unwrap();
    let mut cummulative_dones = [false; N_SNAKES];
    let mut first_turn = true;
    while cummulative_dones.iter().filter(|done| !**done).count() > 0 {
        let turn_direction = if first_turn { direction } else { UP };
        first_turn = false;

        let (rewards, terminations) = game.step([turn_direction; N_SNAKES]).unwrap();
        for (i, termination) in terminations.iter().enumerate() {
            if cummulative_dones[i] {
                assert!(*termination);
                continue;
            }
            if terminations[i] {
                assert!([WIN_REWARD, LOSE_REWARD].contains(&rewards[i]));
                cummulative_dones[i] = true;
            } else {
                assert_eq!(SURVIVE_ROUND_REWARD, rewards[i]);
            }
        }

        let (current_states, snake_facings) = states(&game.game);
        info!(?current_states, ?snake_facings);
        for (i, &done) in cummulative_dones.iter().enumerate() {
            if !done {
                assert_eq!(
                    snake_facings[i], direction,
                    "Snake {i} is not facing {direction}"
                );
            }
        }
    }
}

#[test]
fn run_game_circular_up() {
    run_game_one_direction_circular(UP);
}

#[test]
fn run_game_circular_right() {
    run_game_one_direction_circular(RIGHT);
}

#[test]
fn run_game_circular_down() {
    run_game_one_direction_circular(DOWN);
}

#[test]
fn run_game_circular_left() {
    run_game_one_direction_circular(LEFT);
}

fn run_game_one_direction_circular(direction: isize) {
    init_tracing();
    let mut game = SnakeGame::new().unwrap();
    let mut cummulative_dones = [false; N_SNAKES];
    let mut cummulative_direction = direction;
    while cummulative_dones.iter().filter(|done| !**done).count() > 0 {
        let (rewards, terminations) = game.step([direction; N_SNAKES]).unwrap();
        for (i, termination) in terminations.iter().enumerate() {
            if cummulative_dones[i] {
                assert!(*termination);
                continue;
            }
            if terminations[i] {
                assert!([WIN_REWARD, LOSE_REWARD].contains(&rewards[i]));
                cummulative_dones[i] = true;
            } else {
                assert_eq!(SURVIVE_ROUND_REWARD, rewards[i]);
            }
        }

        let (current_states, snake_facings) = states(&game.game);
        info!(?current_states, ?snake_facings);
        for (i, &done) in cummulative_dones.iter().enumerate() {
            if !done {
                assert_eq!(
                    snake_facings[i], cummulative_direction,
                    "Snake {i} is not facing {cummulative_direction}"
                );
            }
        }

        cummulative_direction = (cummulative_direction + direction).rem_euclid(4);
    }
}

fn init_tracing() {
    _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::builder()
                .with_env_var("LOGLEVEL")
                .with_default_directive(Level::INFO.into())
                .from_env_lossy(),
        )
        .try_init();
}
