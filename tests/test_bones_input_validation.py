import pytest
from scripts.source import safe_float

@pytest.mark.parametrize('row',[{}, {'q':''}, {'q':'nan'}, {'q':'inf'}, {'q':'invalid'}])
def test_missing_or_invalid_motion_values_are_not_zero_filled(row):
    with pytest.raises(ValueError): safe_float(row,'q')
