use super::*;

#[derive(Serialize)]
pub struct Info {
    apiversion: &'static str,
    author: &'static str,
    color: &'static str,
    head: &'static str,
    tail: &'static str,
    version: &'static str,
}

impl Info {
    pub const ME: Info = Info {
        apiversion: "1",
        author: "SichangHe",
        color: "#000000",
        head: "pixel",
        tail: "pixel",
        version: env!("CARGO_PKG_VERSION"),
    };
}
