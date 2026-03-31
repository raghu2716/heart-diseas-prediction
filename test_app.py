import pytest

# Basic test cases for heart disease prediction application

def test_predictions():
    assert predict_heart_disease([1, 2, 1, 1]) in [0, 1]  # Dummy values
    assert predict_heart_disease([2, 1, 1, 0]) in [0, 1]
    assert predict_heart_disease([0, 0, 1, 1]) in [0, 1]

if __name__ == '__main__':
    pytest.main()