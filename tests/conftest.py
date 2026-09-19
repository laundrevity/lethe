def pytest_addoption(parser):
    parser.addoption("--run-sandbox", action="store_true", help="Run real macOS Seatbelt enforcement tests (requires sandbox-exec)")
    parser.addoption("--run-live-tools", action="store_true", help="Test a harmless shell call with the existing model server on port 8080")
