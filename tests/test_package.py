import sentinel_ai


def test_package_can_be_imported() -> None:
    assert sentinel_ai.__name__ == "sentinel_ai"
    assert sentinel_ai.__version__ == "0.1.0"
