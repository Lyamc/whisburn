mod api;
mod app;
mod platform;

pub use app::App;

#[cfg(not(target_arch = "wasm32"))]
pub fn run() -> iced::Result {
    use iced::Theme;
    iced::application(App::title, App::update, App::view)
        .theme(|_| Theme::Light)
        .subscription(App::subscription)
        .window_size(iced::Size::new(720.0, 640.0))
        .run_with(App::new)
}

#[cfg(target_arch = "wasm32")]
pub fn run_web() -> iced::Result {
    use iced::Theme;
    iced::application(App::title, App::update, App::view)
        .theme(|_| Theme::Light)
        .subscription(App::subscription)
        .run_with(App::new)
}