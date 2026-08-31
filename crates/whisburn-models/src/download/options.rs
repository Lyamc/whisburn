#[derive(Debug, Clone)]
pub struct DownloadOptions {
    pub hf_token: Option<String>,
    pub verbose: bool,
    pub force: bool,
}

impl Default for DownloadOptions {
    fn default() -> Self {
        Self {
            hf_token: std::env::var("HF_TOKEN").ok(),
            verbose: false,
            force: false,
        }
    }
}