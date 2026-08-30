fn main() {
    console_error_panic_hook::set_once();
    o3whisburn_app::run_web().expect("failed to start o3whisburn web app");
}