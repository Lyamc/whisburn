fn main() {
    console_error_panic_hook::set_once();
    whisburn_app::run_web().expect("failed to start whisburn web app");
}