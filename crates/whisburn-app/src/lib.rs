mod api;
mod app;
mod platform;

pub use app::App;

fn light_theme(_app: &App) -> iced::Theme {
    iced::Theme::Light
}

#[cfg(not(target_arch = "wasm32"))]
pub fn run() -> iced::Result {
    iced::application(App::new, App::update, App::view)
        .title(App::title)
        .theme(light_theme)
        .subscription(App::subscription)
        .window_size(iced::Size::new(720.0, 640.0))
        .run()
}

#[cfg(target_arch = "wasm32")]
pub fn run_web() -> iced::Result {
    iced::application(App::new, App::update, App::view)
        .title(App::title)
        .theme(light_theme)
        .subscription(App::subscription)
        .run()
}