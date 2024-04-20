use super::*;

#[test]
fn run_game() {
    let mut game = SnakeGame::new().unwrap();
    let (rewards, terminations) = game.step([0; N_SNAKES]).unwrap();
    rewards
        .iter()
        .for_each(|r| assert!([SURVIVE_ROUND_REWARD, LOSE_REWARD].contains(r)));
    assert!(terminations.iter().filter(|t| **t).count() < 3);

    let (states0, snake_facings) = states(&game.game);
    println!("{:?}", states0);
    assert_eq!(states0.len(), N_SNAKES);
    assert_eq!(snake_facings, [0; N_SNAKES]);

    let (rewards, terminations) = game.step([0; N_SNAKES]).unwrap();
    rewards
        .iter()
        .for_each(|r| assert!([SURVIVE_ROUND_REWARD, LOSE_REWARD].contains(r)));
    assert!(terminations.iter().filter(|t| **t).count() < 3);

    let (states1, snake_facings) = states(&game.game);
    println!("{:?}", states1);
    assert_eq!(states1.len(), N_SNAKES);
    assert_eq!(snake_facings, [0; N_SNAKES]);
}
