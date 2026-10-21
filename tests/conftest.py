def pytest_addoption(parser):
    parser.addoption("--run-sandbox", action="store_true", help="Run real macOS Seatbelt enforcement tests (requires sandbox-exec)")
    parser.addoption("--run-live-tools", action="store_true", help="Test a harmless shell call with the existing model server on port 8080")
    parser.addoption("--run-live-media", action="store_true", help="Load a model and test image/video understanding with synthetic fixtures")
    parser.addoption("--media-model", default="unsloth/Qwen3.8-Flash-Next-GGUF:IQ4_XS", help="Model for --run-live-media")
