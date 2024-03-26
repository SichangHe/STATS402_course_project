use battlesnake::*;

fn main() {
    init_info_tracing_from_env();

    let args: Vec<String> = std::env::args().collect();
    run(&args[1]);
}
