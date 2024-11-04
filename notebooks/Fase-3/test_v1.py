import pytest

@pytest.fixture(autouse=True)
def autouse_fixture():
    print("\nExecuting autouse fixture")
    return 5

@pytest.fixture
def normal_fixture(autouse_fixture):
    print("\nExecuting normal fixture")
    return autouse_fixture + 5

def test_example(normal_fixture):
    print("Executing the test")
    assert normal_fixture == 10